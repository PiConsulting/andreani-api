sudo docker build -t andreani-api .
sudo docker tag andreani-api:latest cregwarehousedev.azurecr.io/andreani-api:latest
sudo docker push cregwarehousedev.azurecr.io/andreani-api:latest
