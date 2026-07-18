"""Intelligent cache invalidation service with advanced strategies."""

import asyncio
import fnmatch
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from assistant.config import get_settings
from assistant.core.logging import get_logger
from assistant.infrastructure.cache.advanced_cache_manager import get_cache_manager
from assistant.infrastructure.cache.redis_client import get_redis_client


class InvalidationStrategy(Enum):
    """Cache invalidation strategy enumeration."""

    IMMEDIATE = "immediate"  # Invalidate immediately
    LAZY = "lazy"  # Invalidate on next access
    TIME_BASED = "time_based"  # Invalidate after specific time
    DEPENDENCY_BASED = "dependency"  # Invalidate based on dependencies
    WRITE_THROUGH = "write_through"  # Invalidate and refresh immediately


@dataclass
class InvalidationRule:
    """Cache invalidation rule configuration."""

    pattern: str  # Pattern to match cache keys
    strategy: InvalidationStrategy  # Invalidation strategy
    dependencies: List[str]  # Dependencies that trigger invalidation
    ttl_override: Optional[int]  # Override TTL for this pattern
    priority: int = 0  # Rule priority (higher = more important)
    enabled: bool = True  # Whether rule is active


@dataclass
class InvalidationEvent:
    """Cache invalidation event record."""

    timestamp: datetime
    event_type: str  # Type of event that triggered invalidation
    affected_keys: List[str]  # Keys that were invalidated
    pattern: Optional[str]  # Pattern that matched (if any)
    strategy: InvalidationStrategy  # Strategy used
    duration_ms: float  # Time taken to invalidate
    success: bool  # Whether invalidation succeeded


class CacheInvalidationService:
    """
    Intelligent cache invalidation service.

    Features:
    - Pattern-based invalidation rules
    - Multiple invalidation strategies
    - Dependency tracking and cascade invalidation
    - Performance monitoring and metrics
    - Automatic invalidation based on data changes
    - Smart cache warming after invalidation
    """

    def __init__(self):
        """Initialize the cache invalidation service."""
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)

        # Invalidation rules
        self._rules: List[InvalidationRule] = []
        self._dependency_graph: Dict[str, Set[str]] = {}

        # Event tracking
        self._invalidation_events: List[InvalidationEvent] = []
        self._max_events = 1000  # Keep last 1000 events

        # Performance metrics
        self._metrics = {
            "total_invalidations": 0,
            "successful_invalidations": 0,
            "failed_invalidations": 0,
            "average_invalidation_time_ms": 0.0,
            "rules_applied": 0,
            "dependency_cascades": 0,
        }

        # Initialize default rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Set up default invalidation rules."""
        default_rules = [
            # Chat responses - invalidate when user data changes
            InvalidationRule(
                pattern="chat:*",
                strategy=InvalidationStrategy.IMMEDIATE,
                dependencies=["user_data", "conversation_history"],
                ttl_override=300,
                priority=10,
            ),
            # Knowledge base - invalidate when documents are updated
            InvalidationRule(
                pattern="knowledge:*",
                strategy=InvalidationStrategy.LAZY,
                dependencies=["documents", "embeddings"],
                ttl_override=1800,
                priority=8,
            ),
            # Health endpoints - frequent invalidation
            InvalidationRule(
                pattern="health:*",
                strategy=InvalidationStrategy.TIME_BASED,
                dependencies=[],
                ttl_override=30,
                priority=5,
            ),
            # Database stats - invalidate on schema changes
            InvalidationRule(
                pattern="database:*",
                strategy=InvalidationStrategy.DEPENDENCY_BASED,
                dependencies=["database_schema", "connection_pool"],
                ttl_override=60,
                priority=7,
            ),
            # User sessions - immediate invalidation for security
            InvalidationRule(
                pattern="session:*",
                strategy=InvalidationStrategy.IMMEDIATE,
                dependencies=["user_auth", "permissions"],
                ttl_override=900,
                priority=15,
            ),
            # Static content - rarely invalidate
            InvalidationRule(
                pattern="static:*",
                strategy=InvalidationStrategy.LAZY,
                dependencies=["deployment"],
                ttl_override=86400,
                priority=1,
            ),
        ]

        self._rules.extend(default_rules)
        self.logger.info(f"Initialized {len(default_rules)} default invalidation rules")

    def add_rule(self, rule: InvalidationRule) -> bool:
        """Add a new invalidation rule."""
        try:
            # Validate rule
            if not rule.pattern:
                raise ValueError("Rule pattern cannot be empty")

            # Insert rule in priority order
            inserted = False
            for i, existing_rule in enumerate(self._rules):
                if rule.priority > existing_rule.priority:
                    self._rules.insert(i, rule)
                    inserted = True
                    break

            if not inserted:
                self._rules.append(rule)

            self.logger.info(
                f"Added invalidation rule: {rule.pattern} (strategy: {rule.strategy.value})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to add invalidation rule: {str(e)}")
            return False

    def remove_rule(self, pattern: str) -> bool:
        """Remove invalidation rule by pattern."""
        try:
            original_count = len(self._rules)
            self._rules = [rule for rule in self._rules if rule.pattern != pattern]

            removed_count = original_count - len(self._rules)
            if removed_count > 0:
                self.logger.info(
                    f"Removed {removed_count} invalidation rule(s) for pattern: {pattern}"
                )
                return True
            else:
                self.logger.warning(f"No invalidation rules found for pattern: {pattern}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to remove invalidation rule: {str(e)}")
            return False

    def add_dependency(self, source: str, dependent: str):
        """Add dependency relationship for cascade invalidation."""
        if source not in self._dependency_graph:
            self._dependency_graph[source] = set()

        self._dependency_graph[source].add(dependent)
        self.logger.debug(f"Added dependency: {source} -> {dependent}")

    def _find_matching_rules(self, key_pattern: str) -> List[InvalidationRule]:
        """Find invalidation rules that match the given pattern."""
        matching_rules = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            # Use fnmatch for pattern matching
            if fnmatch.fnmatch(key_pattern, rule.pattern):
                matching_rules.append(rule)

        # Sort by priority (highest first)
        matching_rules.sort(key=lambda r: r.priority, reverse=True)
        return matching_rules

    def _get_dependent_keys(self, dependency: str) -> Set[str]:
        """Get all keys that depend on the given dependency."""
        dependent_keys = set()

        if dependency in self._dependency_graph:
            dependent_keys.update(self._dependency_graph[dependency])

            # Recursive dependency resolution
            for dependent in self._dependency_graph[dependency]:
                dependent_keys.update(self._get_dependent_keys(dependent))

        return dependent_keys

    async def _invalidate_immediate(
        self, cache_names: List[str], key_patterns: List[str]
    ) -> Dict[str, int]:
        """Perform immediate cache invalidation."""
        results = {}

        for cache_name in cache_names:
            try:
                cache_manager = await get_cache_manager(cache_name)
                total_invalidated = 0

                for pattern in key_patterns:
                    # If pattern contains wildcards, we need to get matching keys
                    if "*" in pattern or "?" in pattern:
                        # For now, use tag-based invalidation
                        # In a full implementation, you'd enumerate keys
                        invalidated = await cache_manager.invalidate_by_tags([f"pattern:{pattern}"])
                    else:
                        # Direct key deletion
                        success = await cache_manager.delete(pattern)
                        invalidated = 1 if success else 0

                    total_invalidated += invalidated

                results[cache_name] = total_invalidated

            except Exception as e:
                self.logger.error(f"Failed to invalidate cache {cache_name}: {str(e)}")
                results[cache_name] = 0

        return results

    async def _invalidate_lazy(
        self, cache_names: List[str], key_patterns: List[str]
    ) -> Dict[str, int]:
        """Mark entries for lazy invalidation (invalidate on next access)."""
        results = {}

        # For lazy invalidation, we typically set a special marker
        # that's checked during cache retrieval
        for cache_name in cache_names:
            try:
                redis_client = await get_redis_client()
                if redis_client:
                    total_marked = 0

                    for pattern in key_patterns:
                        # Mark keys as invalid with expiry marker
                        marker_key = f"invalid:{cache_name}:{pattern}"
                        await redis_client.setex(marker_key, 3600, "lazy_invalid")
                        total_marked += 1

                    results[cache_name] = total_marked
                else:
                    results[cache_name] = 0

            except Exception as e:
                self.logger.error(
                    f"Failed to mark cache for lazy invalidation {cache_name}: {str(e)}"
                )
                results[cache_name] = 0

        return results

    async def _invalidate_time_based(
        self, cache_names: List[str], key_patterns: List[str], delay_seconds: int = 0
    ) -> Dict[str, int]:
        """Schedule time-based invalidation."""
        if delay_seconds == 0:
            # Immediate time-based invalidation
            return await self._invalidate_immediate(cache_names, key_patterns)

        # Schedule for later
        async def delayed_invalidation():
            await asyncio.sleep(delay_seconds)
            await self._invalidate_immediate(cache_names, key_patterns)

        # Start the delayed task
        asyncio.create_task(delayed_invalidation())

        # Return estimated count (we don't know actual count until execution)
        return {cache_name: len(key_patterns) for cache_name in cache_names}

    async def _invalidate_write_through(
        self,
        cache_names: List[str],
        key_patterns: List[str],
        refresh_function: Optional[Callable] = None,
    ) -> Dict[str, int]:
        """Invalidate and immediately refresh cache entries."""
        # First invalidate
        invalidation_results = await self._invalidate_immediate(cache_names, key_patterns)

        # Then refresh if function provided
        if refresh_function:
            try:
                for cache_name in cache_names:
                    cache_manager = await get_cache_manager(cache_name)

                    for pattern in key_patterns:
                        if "*" not in pattern and "?" not in pattern:
                            # Direct key refresh
                            new_value = await refresh_function(pattern)
                            if new_value is not None:
                                await cache_manager.set(pattern, new_value)

            except Exception as e:
                self.logger.error(f"Failed to refresh cache after invalidation: {str(e)}")

        return invalidation_results

    async def invalidate_by_pattern(
        self,
        pattern: str,
        cache_names: Optional[List[str]] = None,
        strategy_override: Optional[InvalidationStrategy] = None,
        refresh_function: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Invalidate cache entries by pattern.

        Args:
            pattern: Pattern to match cache keys
            cache_names: Specific cache names to invalidate (default: all)
            strategy_override: Override the strategy from rules
            refresh_function: Function to refresh cache after invalidation

        Returns:
            Invalidation results and statistics
        """
        start_time = time.time()

        try:
            # Find matching rules
            matching_rules = self._find_matching_rules(pattern)

            if not matching_rules and not strategy_override:
                # No rules match, use default immediate strategy
                strategy = InvalidationStrategy.IMMEDIATE
                self.logger.info(f"No rules match pattern '{pattern}', using immediate strategy")
            else:
                # Use highest priority rule or override
                strategy = strategy_override or matching_rules[0].strategy

            # Default cache names if not specified
            if cache_names is None:
                cache_names = ["default", "response_cache", "query_cache"]

            # Execute invalidation based on strategy
            results = {}
            if strategy == InvalidationStrategy.IMMEDIATE:
                results = await self._invalidate_immediate(cache_names, [pattern])
            elif strategy == InvalidationStrategy.LAZY:
                results = await self._invalidate_lazy(cache_names, [pattern])
            elif strategy == InvalidationStrategy.TIME_BASED:
                results = await self._invalidate_time_based(cache_names, [pattern])
            elif strategy == InvalidationStrategy.WRITE_THROUGH:
                results = await self._invalidate_write_through(
                    cache_names, [pattern], refresh_function
                )
            else:
                results = await self._invalidate_immediate(cache_names, [pattern])

            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            total_invalidated = sum(results.values())

            self._metrics["total_invalidations"] += 1
            if total_invalidated > 0:
                self._metrics["successful_invalidations"] += 1
            else:
                self._metrics["failed_invalidations"] += 1

            # Update average time
            if self._metrics["total_invalidations"] > 1:
                self._metrics["average_invalidation_time_ms"] = (
                    self._metrics["average_invalidation_time_ms"]
                    * (self._metrics["total_invalidations"] - 1)
                    + duration_ms
                ) / self._metrics["total_invalidations"]
            else:
                self._metrics["average_invalidation_time_ms"] = duration_ms

            if matching_rules:
                self._metrics["rules_applied"] += 1

            # Record event
            event = InvalidationEvent(
                timestamp=datetime.utcnow(),
                event_type="pattern_invalidation",
                affected_keys=[pattern],
                pattern=pattern,
                strategy=strategy,
                duration_ms=duration_ms,
                success=total_invalidated > 0,
            )
            self._add_event(event)

            self.logger.info(
                f"Invalidated pattern '{pattern}' using {strategy.value} strategy: "
                f"{total_invalidated} entries in {duration_ms:.2f}ms"
            )

            return {
                "pattern": pattern,
                "strategy": strategy.value,
                "cache_results": results,
                "total_invalidated": total_invalidated,
                "duration_ms": duration_ms,
                "rules_matched": len(matching_rules),
                "success": total_invalidated > 0,
            }

        except Exception as e:
            self.logger.error(f"Failed to invalidate pattern '{pattern}': {str(e)}")
            return {"pattern": pattern, "error": str(e), "success": False}

    async def invalidate_by_dependency(
        self, dependency: str, cascade: bool = True
    ) -> Dict[str, Any]:
        """
        Invalidate cache entries based on dependency changes.

        Args:
            dependency: Dependency that changed
            cascade: Whether to cascade to dependent items

        Returns:
            Invalidation results
        """
        start_time = time.time()

        try:
            invalidated_patterns = set()

            # Find rules that depend on this dependency
            dependent_rules = [
                rule for rule in self._rules if dependency in rule.dependencies and rule.enabled
            ]

            # Invalidate matching patterns
            for rule in dependent_rules:
                result = await self.invalidate_by_pattern(
                    rule.pattern, strategy_override=rule.strategy
                )
                if result.get("success", False):
                    invalidated_patterns.add(rule.pattern)

            # Handle cascade invalidation
            cascade_count = 0
            if cascade:
                dependent_keys = self._get_dependent_keys(dependency)
                for dependent_key in dependent_keys:
                    cascade_result = await self.invalidate_by_dependency(
                        dependent_key, cascade=False
                    )
                    if cascade_result.get("success", False):
                        cascade_count += 1

            if cascade_count > 0:
                self._metrics["dependency_cascades"] += 1

            duration_ms = (time.time() - start_time) * 1000

            # Record event
            event = InvalidationEvent(
                timestamp=datetime.utcnow(),
                event_type="dependency_invalidation",
                affected_keys=list(invalidated_patterns),
                pattern=None,
                strategy=InvalidationStrategy.DEPENDENCY_BASED,
                duration_ms=duration_ms,
                success=len(invalidated_patterns) > 0,
            )
            self._add_event(event)

            self.logger.info(
                f"Dependency '{dependency}' invalidation: "
                f"{len(invalidated_patterns)} patterns, {cascade_count} cascades, {duration_ms:.2f}ms"
            )

            return {
                "dependency": dependency,
                "patterns_invalidated": list(invalidated_patterns),
                "cascade_count": cascade_count,
                "duration_ms": duration_ms,
                "success": len(invalidated_patterns) > 0 or cascade_count > 0,
            }

        except Exception as e:
            self.logger.error(f"Failed to invalidate dependency '{dependency}': {str(e)}")
            return {"dependency": dependency, "error": str(e), "success": False}

    def _add_event(self, event: InvalidationEvent):
        """Add invalidation event to history."""
        self._invalidation_events.append(event)

        # Keep only recent events
        if len(self._invalidation_events) > self._max_events:
            self._invalidation_events = self._invalidation_events[-self._max_events :]

    async def get_invalidation_stats(self) -> Dict[str, Any]:
        """Get comprehensive invalidation statistics."""
        try:
            # Calculate recent activity
            recent_events = [
                event
                for event in self._invalidation_events
                if (datetime.utcnow() - event.timestamp).total_seconds() < 3600  # Last hour
            ]

            successful_recent = sum(1 for event in recent_events if event.success)
            failed_recent = len(recent_events) - successful_recent

            return {
                "service_name": "CacheInvalidationService",
                "total_rules": len(self._rules),
                "active_rules": sum(1 for rule in self._rules if rule.enabled),
                "dependency_relationships": len(self._dependency_graph),
                "performance_metrics": self._metrics.copy(),
                "recent_activity": {
                    "last_hour_events": len(recent_events),
                    "last_hour_successful": successful_recent,
                    "last_hour_failed": failed_recent,
                    "last_hour_success_rate": (
                        successful_recent / len(recent_events) if recent_events else 0
                    ),
                },
                "rules_summary": [
                    {
                        "pattern": rule.pattern,
                        "strategy": rule.strategy.value,
                        "priority": rule.priority,
                        "enabled": rule.enabled,
                        "dependencies": rule.dependencies,
                    }
                    for rule in self._rules
                ],
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get invalidation stats: {str(e)}")
            return {"error": str(e)}

    async def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent invalidation events."""
        try:
            recent_events = self._invalidation_events[-limit:]
            return [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "affected_keys": event.affected_keys,
                    "pattern": event.pattern,
                    "strategy": event.strategy.value,
                    "duration_ms": event.duration_ms,
                    "success": event.success,
                }
                for event in recent_events
            ]

        except Exception as e:
            self.logger.error(f"Failed to get recent events: {str(e)}")
            return []


# Global invalidation service instance
_invalidation_service: Optional[CacheInvalidationService] = None


async def get_invalidation_service() -> CacheInvalidationService:
    """Get the global cache invalidation service."""
    global _invalidation_service

    if _invalidation_service is None:
        _invalidation_service = CacheInvalidationService()

    return _invalidation_service
