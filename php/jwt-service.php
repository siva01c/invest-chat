<?php

namespace App;

class JWTService
{
    private string $secretKey;
    private string $algorithm;

    public function __construct(string $secretKey, string $algorithm = 'HS256')
    {
        $this->secretKey = $secretKey;
        $this->algorithm = $algorithm;
    }

    private function base64UrlEncode(string $data): string
    {
        return rtrim(strtr(base64_encode($data), '+/', '-_'), '=');
    }

    private function base64UrlDecode(string $data): string
    {
        $pad = strlen($data) % 4;
        if ($pad) {
            $data .= str_repeat('=', 4 - $pad);
        }
        return base64_decode(strtr($data, '-_', '+/'));
    }

    public function generateToken(array $payload, int $expirationMinutes = 30): string
    {
        $header = [
            'typ' => 'JWT',
            'alg' => $this->algorithm
        ];

        $payload['exp'] = time() + ($expirationMinutes * 60);
        
        $headerEncoded = $this->base64UrlEncode(json_encode($header));
        $payloadEncoded = $this->base64UrlEncode(json_encode($payload));
        
        $signature = hash_hmac(
            'sha256',
            "$headerEncoded.$payloadEncoded",
            $this->secretKey,
            true
        );
        
        $signatureEncoded = $this->base64UrlEncode($signature);
        
        return "$headerEncoded.$payloadEncoded.$signatureEncoded";
    }
}


if (php_sapi_name() === 'cli') {
    $secretKey = '152AWESQE_weqew-WEQR5';
    $jwtService = new JWTService($secretKey);

    $payload = ['user_id' => 'ludekkvapil'];
    $token = $jwtService->generateToken($payload);
    echo "Generated Token: " . $token . PHP_EOL;
}