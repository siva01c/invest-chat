import jwt
import datetime

class JWTService:
    def __init__(self, secret_key, algorithm='HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def generate_token(self, payload, expiration_minutes=30):
        payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(minutes=expiration_minutes)
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token

    def validate_token(self, token):
        try:
            decoded_token = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return decoded_token
        except jwt.ExpiredSignatureError:
            return 'Token has expired'
        except jwt.InvalidTokenError:
            return 'Invalid token'

# Example usage:
if __name__ == "__main__":
    secret_key = '152AWESQE_weqew-WEQR5'
    jwt_service = JWTService(secret_key)

    payload = {'user_id': "ludekkvapil"}
    token = jwt_service.generate_token(payload)
    print(f"Generated Token: {token}")

    decoded_payload = jwt_service.validate_token('eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoibHVkZWtrdmFwaWwiLCJleHAiOjE3NDE4NzIwNTV9.-fbafdQnjCOpeKeClmGc3jDZipdqvK2tyXiKhf3BAIY')
    print(f"Decoded Payload: {decoded_payload}")