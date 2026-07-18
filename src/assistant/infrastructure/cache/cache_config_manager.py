"""Cache configuration and TTL policy management system."""

import fnmatch
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from assistant.config import get_settings
from assistant.core.logging import get_logger


class TTLPolicy(Enum):
    """TTL policy types."""

    FIXED = "fixed"  # Fixed TTL for all entries
    SLIDING = "sliding"  # TTL extends on access
    ADAPTIVE = "adaptive"  # TTL adjusts based on usage patterns
    CONTENT_BASED = "content_based"  # TTL based on content characteristics
    TIME_OF_DAY = "time_of_day"  # TTL varies by time of day
    LOAD_BASED = "load_based"  # TTL adjusts based on system load


@dataclass
class CacheProfile:
    """Cache configuration profile for different use cases."""

    name: str
    description: str
    max_memory_mb: int = 100
    default_ttl_seconds: int = 3600
    ttl_policy: TTLPolicy = TTLPolicy.FIXED
    compression_enabled: bool = False
    metrics_enabled: bool = True
    eviction_strategy: str = "lru"

    # TTL policy specific settings
    sliding_window_seconds: int = 300
    adaptive_min_ttl: int = 60
    adaptive_max_ttl: int = 7200
    load_threshold: float = 0.8

    # Content-based TTL rules
    content_size_multiplier: float = 1.0
    json_ttl_factor: float = 1.0
    text_ttl_factor: float = 1.2

    # Time-based TTL rules
    peak_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 14, 15, 16])
    peak_ttl_factor: float = 0.5  # Shorter TTL during peak hours
    off_peak_ttl_factor: float = 2.0  # Longer TTL during off-peak


@dataclass
class CacheRule:
    """Individual cache rule configuration."""

    pattern: str  # Pattern to match cache keys
    profile_name: str  # Profile to apply
    ttl_override: Optional[int] = None  # Override TTL for this rule
    priority: int = 0  # Rule priority (higher = more important)
    enabled: bool = True  # Whether rule is active
    tags: List[str] = field(default_factory=list)  # Tags for grouping

    # Conditional rules
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Custom functions
    ttl_calculator: Optional[str] = None  # Name of custom TTL calculation function


class CacheConfigManager:
    """
    Advanced cache configuration and TTL policy manager.

    Features:
    - Multiple cache profiles for different use cases
    - Dynamic TTL calculation based on various factors
    - Rule-based cache configuration
    - Performance monitoring and auto-tuning
    - Hot-reload of configuration changes
    """

    def __init__(self):
        """Initialize the cache configuration manager."""
        self.settings = get_settings()
        self.logger = get_logger(self.__class__.__name__)

        # Configuration storage
        self._profiles: Dict[str, CacheProfile] = {}
        self._rules: List[CacheRule] = []
        self._custom_ttl_functions: Dict[str, Callable] = {}

        # Performance tracking
        self._performance_data: Dict[str, Dict[str, Any]] = {}
        self._auto_tuning_enabled = True

        # Configuration file paths
        self._config_file = "cache_config.json"

        # Initialize default profiles and rules
        self._setup_default_profiles()
        self._setup_default_rules()
        self._register_default_ttl_functions()

    def _setup_default_profiles(self):
        """Set up default cache profiles."""
        profiles = {
            "high_performance": CacheProfile(
                name="high_performance",
                description="High-performance caching for frequently accessed data",
                max_memory_mb=500,
                default_ttl_seconds=300,
                ttl_policy=TTLPolicy.SLIDING,
                compression_enabled=False,
                eviction_strategy="lru",
                sliding_window_seconds=60,
            ),
            "memory_efficient": CacheProfile(
                name="memory_efficient",
                description="Memory-efficient caching with compression",
                max_memory_mb=100,
                default_ttl_seconds=1800,
                ttl_policy=TTLPolicy.ADAPTIVE,
                compression_enabled=True,
                eviction_strategy="lfu",
                adaptive_min_ttl=300,
                adaptive_max_ttl=3600,
            ),
            "long_term": CacheProfile(
                name="long_term",
                description="Long-term caching for stable data",
                max_memory_mb=200,
                default_ttl_seconds=86400,  # 24 hours
                ttl_policy=TTLPolicy.FIXED,
                compression_enabled=True,
                eviction_strategy="ttl",
            ),
            "session_based": CacheProfile(
                name="session_based",
                description="Session-based caching with sliding expiration",
                max_memory_mb=150,
                default_ttl_seconds=1800,  # 30 minutes
                ttl_policy=TTLPolicy.SLIDING,
                compression_enabled=False,
                eviction_strategy="lru",
                sliding_window_seconds=300,
            ),
            "content_adaptive": CacheProfile(
                name="content_adaptive",
                description="Content-aware caching with adaptive TTL",
                max_memory_mb=300,
                default_ttl_seconds=3600,
                ttl_policy=TTLPolicy.CONTENT_BASED,
                compression_enabled=True,
                eviction_strategy="lru",
                content_size_multiplier=0.5,
                json_ttl_factor=1.0,
                text_ttl_factor=1.5,
            ),
            "load_balanced": CacheProfile(
                name="load_balanced",
                description="Load-aware caching that adjusts based on system load",
                max_memory_mb=250,
                default_ttl_seconds=1800,
                ttl_policy=TTLPolicy.LOAD_BASED,
                compression_enabled=True,
                eviction_strategy="lru",
                load_threshold=0.7,
            ),
        }

        self._profiles.update(profiles)
        self.logger.info(f"Initialized {len(profiles)} default cache profiles")

    def _setup_default_rules(self):
        """Set up default cache rules."""
        rules = [
            # Chat responses - high performance, short TTL
            CacheRule(
                pattern="chat:*",
                profile_name="session_based",
                ttl_override=300,
                priority=10,
                tags=["chat", "user_data"],
            ),
            # Health endpoints - very short TTL
            CacheRule(
                pattern="health:*",
                profile_name="high_performance",
                ttl_override=30,
                priority=8,
                tags=["health", "monitoring"],
            ),
            # Knowledge base - long term with content adaptation
            CacheRule(
                pattern="knowledge:*",
                profile_name="content_adaptive",
                priority=7,
                tags=["knowledge", "search"],
            ),
            # Database stats - load-based caching
            CacheRule(
                pattern="database:*",
                profile_name="load_balanced",
                ttl_override=60,
                priority=6,
                tags=["database", "stats"],
            ),
            # User sessions - secure, sliding expiration
            CacheRule(
                pattern="session:*",
                profile_name="session_based",
                ttl_override=1800,
                priority=15,
                tags=["session", "auth", "security"],
            ),
            # Static content - long term
            CacheRule(
                pattern="static:*",
                profile_name="long_term",
                ttl_override=86400,
                priority=1,
                tags=["static", "assets"],
            ),
            # API responses - adaptive based on endpoint
            CacheRule(
                pattern="response:*",
                profile_name="memory_efficient",
                priority=5,
                tags=["response", "api"],
            ),
            # Performance metrics - frequent updates
            CacheRule(
                pattern="metrics:*",
                profile_name="high_performance",
                ttl_override=15,
                priority=9,
                tags=["metrics", "performance"],
            ),
        ]

        self._rules.extend(rules)
        self.logger.info(f"Initialized {len(rules)} default cache rules")

    def _register_default_ttl_functions(self):
        """Register default TTL calculation functions."""

        def content_size_ttl(content: Any, base_ttl: int, profile: CacheProfile) -> int:
            """Calculate TTL based on content size."""
            try:
                if isinstance(content, (str, bytes)):
                    size = len(content)
                elif isinstance(content, (dict, list)):
                    size = len(json.dumps(content))
                else:
                    size = len(str(content))

                # Larger content gets longer TTL (up to a point)
                size_factor = min(2.0, 1.0 + (size / 10000) * profile.content_size_multiplier)
                return int(base_ttl * size_factor)
            except Exception:
                return base_ttl

        def time_of_day_ttl(content: Any, base_ttl: int, profile: CacheProfile) -> int:
            """Calculate TTL based on time of day."""
            current_hour = datetime.now().hour

            if current_hour in profile.peak_hours:
                return int(base_ttl * profile.peak_ttl_factor)
            else:
                return int(base_ttl * profile.off_peak_ttl_factor)

        def load_based_ttl(content: Any, base_ttl: int, profile: CacheProfile) -> int:
            """Calculate TTL based on system load."""
            try:
                import psutil

                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent

                avg_load = (cpu_percent + memory_percent) / 200.0  # Normalize to 0-1

                if avg_load > profile.load_threshold:
                    # High load - shorter TTL to reduce cache pressure
                    return int(base_ttl * 0.5)
                else:
                    # Low load - normal or longer TTL
                    return int(base_ttl * 1.5)
            except ImportError:
                return base_ttl
            except Exception:
                return base_ttl

        def adaptive_ttl(content: Any, base_ttl: int, profile: CacheProfile) -> int:
            """Calculate adaptive TTL based on access patterns."""
            # This would typically use historical access data
            # For now, return a value between min and max
            return max(profile.adaptive_min_ttl, min(profile.adaptive_max_ttl, base_ttl))

        # Register functions
        self._custom_ttl_functions.update(
            {
                "content_size_ttl": content_size_ttl,
                "time_of_day_ttl": time_of_day_ttl,
                "load_based_ttl": load_based_ttl,
                "adaptive_ttl": adaptive_ttl,
            }
        )

        self.logger.info(f"Registered {len(self._custom_ttl_functions)} TTL calculation functions")

    def add_profile(self, profile: CacheProfile) -> bool:
        """Add a new cache profile."""
        try:
            self._profiles[profile.name] = profile
            self.logger.info(f"Added cache profile: {profile.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add cache profile: {str(e)}")
            return False

    def add_rule(self, rule: CacheRule) -> bool:
        """Add a new cache rule."""
        try:
            # Validate profile exists
            if rule.profile_name not in self._profiles:
                raise ValueError(f"Profile '{rule.profile_name}' does not exist")

            # Insert in priority order
            inserted = False
            for i, existing_rule in enumerate(self._rules):
                if rule.priority > existing_rule.priority:
                    self._rules.insert(i, rule)
                    inserted = True
                    break

            if not inserted:
                self._rules.append(rule)

            self.logger.info(f"Added cache rule: {rule.pattern} -> {rule.profile_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add cache rule: {str(e)}")
            return False

    def register_ttl_function(self, name: str, function: Callable) -> bool:
        """Register a custom TTL calculation function."""
        try:
            self._custom_ttl_functions[name] = function
            self.logger.info(f"Registered TTL function: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register TTL function: {str(e)}")
            return False

    def get_cache_config(self, cache_key: str) -> Dict[str, Any]:
        """Get complete cache configuration for a key."""
        try:
            # Find matching rule
            matching_rule = None
            for rule in self._rules:
                if rule.enabled and fnmatch.fnmatch(cache_key, rule.pattern):
                    matching_rule = rule
                    break

            if not matching_rule:
                # Use default profile
                profile = self._profiles.get("memory_efficient")
                return {
                    "profile": profile,
                    "rule": None,
                    "ttl_seconds": profile.default_ttl_seconds if profile else 3600,
                    "error": "No matching rule found, using default profile",
                }

            # Get profile
            profile = self._profiles.get(matching_rule.profile_name)
            if not profile:
                return {
                    "profile": None,
                    "rule": matching_rule,
                    "ttl_seconds": 3600,
                    "error": f"Profile '{matching_rule.profile_name}' not found",
                }

            # Calculate TTL
            base_ttl = matching_rule.ttl_override or profile.default_ttl_seconds
            final_ttl = self._calculate_dynamic_ttl(None, base_ttl, profile, matching_rule)

            return {
                "profile": profile,
                "rule": matching_rule,
                "ttl_seconds": final_ttl,
                "base_ttl": base_ttl,
                "ttl_policy": profile.ttl_policy.value,
                "cache_key": cache_key,
            }

        except Exception as e:
            self.logger.error(f"Failed to get cache config for key '{cache_key}': {str(e)}")
            return {"profile": None, "rule": None, "ttl_seconds": 3600, "error": str(e)}

    def _calculate_dynamic_ttl(
        self, content: Any, base_ttl: int, profile: CacheProfile, rule: CacheRule
    ) -> int:
        """Calculate dynamic TTL based on policy and content."""
        try:
            # Use custom TTL function if specified
            if rule.ttl_calculator and rule.ttl_calculator in self._custom_ttl_functions:
                return self._custom_ttl_functions[rule.ttl_calculator](content, base_ttl, profile)

            # Apply TTL policy
            if profile.ttl_policy == TTLPolicy.FIXED:
                return base_ttl

            elif profile.ttl_policy == TTLPolicy.SLIDING:
                # For sliding, we return base TTL but the cache manager will extend on access
                return base_ttl

            elif profile.ttl_policy == TTLPolicy.ADAPTIVE:
                return self._custom_ttl_functions["adaptive_ttl"](content, base_ttl, profile)

            elif profile.ttl_policy == TTLPolicy.CONTENT_BASED:
                return self._custom_ttl_functions["content_size_ttl"](content, base_ttl, profile)

            elif profile.ttl_policy == TTLPolicy.TIME_OF_DAY:
                return self._custom_ttl_functions["time_of_day_ttl"](content, base_ttl, profile)

            elif profile.ttl_policy == TTLPolicy.LOAD_BASED:
                return self._custom_ttl_functions["load_based_ttl"](content, base_ttl, profile)

            else:
                return base_ttl

        except Exception as e:
            self.logger.warning(f"Failed to calculate dynamic TTL: {str(e)}")
            return base_ttl

    def calculate_ttl(self, cache_key: str, content: Any = None) -> int:
        """Calculate TTL for a specific cache key and content."""
        config = self.get_cache_config(cache_key)

        if config.get("error"):
            return config["ttl_seconds"]

        profile = config["profile"]
        rule = config["rule"]
        base_ttl = config["base_ttl"]

        return self._calculate_dynamic_ttl(content, base_ttl, profile, rule)

    async def update_performance_data(self, cache_key: str, metrics: Dict[str, Any]):
        """Update performance data for auto-tuning."""
        if not self._auto_tuning_enabled:
            return

        try:
            pattern = self._find_pattern_for_key(cache_key)
            if pattern:
                if pattern not in self._performance_data:
                    self._performance_data[pattern] = {
                        "hit_rate": 0.0,
                        "average_ttl": 0,
                        "access_frequency": 0,
                        "last_updated": datetime.utcnow(),
                        "sample_count": 0,
                    }

                perf_data = self._performance_data[pattern]

                # Update metrics with exponential moving average
                alpha = 0.1  # Smoothing factor
                perf_data["hit_rate"] = (
                    alpha * metrics.get("hit_rate", 0) + (1 - alpha) * perf_data["hit_rate"]
                )

                perf_data["sample_count"] += 1
                perf_data["last_updated"] = datetime.utcnow()

                # Auto-tune TTL based on performance
                await self._auto_tune_ttl(pattern, perf_data)

        except Exception as e:
            self.logger.warning(f"Failed to update performance data: {str(e)}")

    def _find_pattern_for_key(self, cache_key: str) -> Optional[str]:
        """Find the pattern that matches a cache key."""
        for rule in self._rules:
            if rule.enabled and fnmatch.fnmatch(cache_key, rule.pattern):
                return rule.pattern
        return None

    async def _auto_tune_ttl(self, pattern: str, perf_data: Dict[str, Any]):
        """Auto-tune TTL based on performance data."""
        try:
            hit_rate = perf_data["hit_rate"]

            # Find rule for this pattern
            rule = next((r for r in self._rules if r.pattern == pattern), None)
            if not rule:
                return

            # Adjust TTL based on hit rate
            if hit_rate < 0.5:
                # Low hit rate - increase TTL
                if rule.ttl_override:
                    rule.ttl_override = min(rule.ttl_override * 1.2, 86400)  # Cap at 24 hours
                    self.logger.info(
                        f"Auto-tuned TTL for pattern '{pattern}': increased to {rule.ttl_override}s"
                    )

            elif hit_rate > 0.9:
                # Very high hit rate - might be able to reduce TTL
                if rule.ttl_override and rule.ttl_override > 60:
                    rule.ttl_override = max(rule.ttl_override * 0.9, 60)  # Min 1 minute
                    self.logger.info(
                        f"Auto-tuned TTL for pattern '{pattern}': decreased to {rule.ttl_override}s"
                    )

        except Exception as e:
            self.logger.warning(f"Failed to auto-tune TTL for pattern '{pattern}': {str(e)}")

    async def save_config(self, filename: Optional[str] = None) -> bool:
        """Save current configuration to file."""
        try:
            config_file = filename or self._config_file

            config_data = {
                "profiles": {
                    name: {
                        "name": profile.name,
                        "description": profile.description,
                        "max_memory_mb": profile.max_memory_mb,
                        "default_ttl_seconds": profile.default_ttl_seconds,
                        "ttl_policy": profile.ttl_policy.value,
                        "compression_enabled": profile.compression_enabled,
                        "metrics_enabled": profile.metrics_enabled,
                        "eviction_strategy": profile.eviction_strategy,
                        "sliding_window_seconds": profile.sliding_window_seconds,
                        "adaptive_min_ttl": profile.adaptive_min_ttl,
                        "adaptive_max_ttl": profile.adaptive_max_ttl,
                        "load_threshold": profile.load_threshold,
                        "content_size_multiplier": profile.content_size_multiplier,
                        "json_ttl_factor": profile.json_ttl_factor,
                        "text_ttl_factor": profile.text_ttl_factor,
                        "peak_hours": profile.peak_hours,
                        "peak_ttl_factor": profile.peak_ttl_factor,
                        "off_peak_ttl_factor": profile.off_peak_ttl_factor,
                    }
                    for name, profile in self._profiles.items()
                },
                "rules": [
                    {
                        "pattern": rule.pattern,
                        "profile_name": rule.profile_name,
                        "ttl_override": rule.ttl_override,
                        "priority": rule.priority,
                        "enabled": rule.enabled,
                        "tags": rule.tags,
                        "conditions": rule.conditions,
                        "ttl_calculator": rule.ttl_calculator,
                    }
                    for rule in self._rules
                ],
                "performance_data": self._performance_data,
                "auto_tuning_enabled": self._auto_tuning_enabled,
                "saved_at": datetime.utcnow().isoformat(),
            }

            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)

            self.logger.info(f"Saved cache configuration to {config_file}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save cache configuration: {str(e)}")
            return False

    async def load_config(self, filename: Optional[str] = None) -> bool:
        """Load configuration from file."""
        try:
            config_file = filename or self._config_file

            with open(config_file, "r") as f:
                config_data = json.load(f)

            # Load profiles
            for name, profile_data in config_data.get("profiles", {}).items():
                profile = CacheProfile(
                    name=profile_data["name"],
                    description=profile_data["description"],
                    max_memory_mb=profile_data["max_memory_mb"],
                    default_ttl_seconds=profile_data["default_ttl_seconds"],
                    ttl_policy=TTLPolicy(profile_data["ttl_policy"]),
                    compression_enabled=profile_data["compression_enabled"],
                    metrics_enabled=profile_data["metrics_enabled"],
                    eviction_strategy=profile_data["eviction_strategy"],
                    sliding_window_seconds=profile_data.get("sliding_window_seconds", 300),
                    adaptive_min_ttl=profile_data.get("adaptive_min_ttl", 60),
                    adaptive_max_ttl=profile_data.get("adaptive_max_ttl", 7200),
                    load_threshold=profile_data.get("load_threshold", 0.8),
                    content_size_multiplier=profile_data.get("content_size_multiplier", 1.0),
                    json_ttl_factor=profile_data.get("json_ttl_factor", 1.0),
                    text_ttl_factor=profile_data.get("text_ttl_factor", 1.2),
                    peak_hours=profile_data.get("peak_hours", [9, 10, 11, 14, 15, 16]),
                    peak_ttl_factor=profile_data.get("peak_ttl_factor", 0.5),
                    off_peak_ttl_factor=profile_data.get("off_peak_ttl_factor", 2.0),
                )
                self._profiles[name] = profile

            # Load rules
            self._rules.clear()
            for rule_data in config_data.get("rules", []):
                rule = CacheRule(
                    pattern=rule_data["pattern"],
                    profile_name=rule_data["profile_name"],
                    ttl_override=rule_data.get("ttl_override"),
                    priority=rule_data.get("priority", 0),
                    enabled=rule_data.get("enabled", True),
                    tags=rule_data.get("tags", []),
                    conditions=rule_data.get("conditions", {}),
                    ttl_calculator=rule_data.get("ttl_calculator"),
                )
                self._rules.append(rule)

            # Sort rules by priority
            self._rules.sort(key=lambda r: r.priority, reverse=True)

            # Load performance data
            self._performance_data = config_data.get("performance_data", {})
            self._auto_tuning_enabled = config_data.get("auto_tuning_enabled", True)

            self.logger.info(f"Loaded cache configuration from {config_file}")
            return True

        except FileNotFoundError:
            self.logger.info(f"Configuration file {config_file} not found, using defaults")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load cache configuration: {str(e)}")
            return False

    async def get_config_summary(self) -> Dict[str, Any]:
        """Get comprehensive configuration summary."""
        try:
            return {
                "profiles": {
                    name: {
                        "description": profile.description,
                        "max_memory_mb": profile.max_memory_mb,
                        "default_ttl_seconds": profile.default_ttl_seconds,
                        "ttl_policy": profile.ttl_policy.value,
                        "compression_enabled": profile.compression_enabled,
                        "eviction_strategy": profile.eviction_strategy,
                    }
                    for name, profile in self._profiles.items()
                },
                "rules": [
                    {
                        "pattern": rule.pattern,
                        "profile_name": rule.profile_name,
                        "ttl_override": rule.ttl_override,
                        "priority": rule.priority,
                        "enabled": rule.enabled,
                        "tags": rule.tags,
                    }
                    for rule in self._rules
                ],
                "ttl_functions": list(self._custom_ttl_functions.keys()),
                "auto_tuning_enabled": self._auto_tuning_enabled,
                "performance_patterns": len(self._performance_data),
                "total_profiles": len(self._profiles),
                "total_rules": len(self._rules),
                "active_rules": sum(1 for rule in self._rules if rule.enabled),
            }

        except Exception as e:
            self.logger.error(f"Failed to get config summary: {str(e)}")
            return {"error": str(e)}


# Global configuration manager instance
_config_manager: Optional[CacheConfigManager] = None


async def get_cache_config_manager() -> CacheConfigManager:
    """Get the global cache configuration manager."""
    global _config_manager

    if _config_manager is None:
        _config_manager = CacheConfigManager()
        await _config_manager.load_config()

    return _config_manager
