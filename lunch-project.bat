@echo off
python FedAvg.py --local_epochs 1 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python FedAvg.py --local_epochs 5 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python FedAvg.py --local_epochs 10 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%

python Fedprox.py --local_epochs 1 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python Fedprox.py --local_epochs 5 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python Fedprox.py --local_epochs 10 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%

python Fedper.py --local_epochs 1 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python Fedper.py --local_epochs 5 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python Fedper.py --local_epochs 10 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%


python FedMAML.py --local_epochs 1 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python FedMAML.py --local_epochs 5 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python FedMAML.py --local_epochs 10 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%

python FedGA --local_epochs 1 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python FedGA --local_epochs 5 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python FedGA --local_epochs 10 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%

python FedGA_Meta --local_epochs 1 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python FedGA_Meta --local_epochs 5 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%
python FedGA_Meta --local_epochs 10 --epochs 52
if %errorlevel% neq 0 exit /b %errorlevel%

pause