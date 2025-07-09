# jwt_handler.py
import httpx
import json
import os
from typing import Optional
from abc import ABC, abstractmethod

class Consumer(ABC):
    @abstractmethod
    async def get_api(self):
        pass

class DrupalConsumer(Consumer):
    def __init__(self, base_url: str = None, username: str = None, password: str = None):
        self.base_url = base_url or os.getenv('DRUPAL_BASE_URL', 'http://drupal.ddev.site')
        self.username = username or os.getenv('DRUPAL_USERNAME', 'api')
        self.password = password or os.getenv('DRUPAL_PASSWORD', '123456')

    async def _get_csrf_token(self) -> Optional[str]:
        """Fetch CSRF token required for authentication."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f'{self.base_url}/session/token')
                response.raise_for_status()
                return response.text
            except httpx.RequestError as e:
                print(f"Error fetching CSRF token: {e}")
                return None

    async def _login(self) -> Optional[str]:
        """Login and retrieve access token."""
        csrf_token = await self._get_csrf_token()
        if not csrf_token:
            return None  # Prevent login attempt without CSRF token
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f'{self.base_url}/user/login?_format=json',
                    headers={'Content-Type': 'application/json', 'X-CSRF-Token': csrf_token},
                    json={"name": self.username, "pass": self.password},
                )
                response.raise_for_status()
                data = response.json()
                return data.get("access_token")
            except httpx.RequestError as e:
                print(f"Error during login: {e}")
                return None

    async def _get_jwt_token(self) -> Optional[str]:
        """Retrieve a JWT token using the access token."""
        access_token = await self._login()
        if not access_token:
            return None  # Prevent request without an access token
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f'{self.base_url}/jwt/token',
                    headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {access_token}'},
                )
                response.raise_for_status()
                data = response.json()
                return data.get("token")
            except httpx.RequestError as e:
                print(f"Error fetching Drupal token: {e}")
                return None

    async def get_api(self):
        jwt_token = await self._get_jwt_token()
        if not jwt_token:
            return None
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f'{self.base_url}/api',
                    headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {jwt_token}'},
                )
                response.raise_for_status()
                data = response.json()
                return data
            except httpx.RequestError as e:
                print(f"Error fetching Drupal token: {e}")
                return None 