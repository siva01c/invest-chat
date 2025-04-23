## RSA

'''
openssl genrsa 2048 >  jwt.key.txt


openssl rsa -in  jwt.key.txt  -pubout
'''


## HMAC key 

head -c 64 /dev/urandom | base64 -w 0 > /path/to/private/dir/jwt.key.txt
