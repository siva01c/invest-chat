ps aux | grep api_server.py 


EC2 user data
#!/bin/bash
cd /home/ec2-user/apiserver
sudo systemctl start nginx
nohup python3 api_server.py > logs/output.log 2>&1 &



ssh  -i  ~/.ssh/api-server.pem ec2-user@18.197.124.128

