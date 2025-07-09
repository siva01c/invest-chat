# API Documentation

This directory contains comprehensive API documentation for the Sales Assistant system.

## API Overview

The Sales Assistant provides a RESTful API for interacting with the AI chatbot system. The API is built using FastAPI and provides both synchronous and asynchronous endpoints.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## API Documentation

### 📋 Core Endpoints
- **[REST API](rest-api.md)** - Complete REST API reference
- **[WebSocket API](websocket.md)** - Real-time chat via WebSocket
- **[Authentication](authentication.md)** - API authentication methods
- **[Rate Limiting](rate-limiting.md)** - Rate limiting policies

### 🔧 Interactive Documentation

When running the application, you can access interactive API documentation at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Quick Start

### Basic Chat Request

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, I would like to know about Luděk'\''s services"
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Response Format

All API responses follow a consistent format:

```json
{
  "status": "success",
  "data": {
    "message": "Response from AI assistant",
    "session_id": "abc123",
    "timestamp": "2024-01-15T10:30:00Z"
  },
  "error": null
}
```

## Error Handling

Errors are returned with appropriate HTTP status codes:

- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Too Many Requests
- `500` - Internal Server Error

## SDKs and Libraries

### Python SDK

```python
from sales_assistant.client import SalesAssistantClient

client = SalesAssistantClient("http://localhost:8000")
response = client.chat("Hello, tell me about Luděk's experience")
print(response.message)
```

### JavaScript SDK

```javascript
import { SalesAssistantClient } from 'sales-assistant-js';

const client = new SalesAssistantClient('http://localhost:8000');
const response = await client.chat("Hello, tell me about Luděk's experience");
console.log(response.message);
```

## Support

For API support:
- Check the [FAQ](../user/faq.md)
- Review the [User Guide](../user/user-guide.md)
- Contact: info@ludekkvapil.cz