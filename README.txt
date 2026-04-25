This program must use an NVIDA GPU and Windows.
You has to install python version 3.10 and add it into your environment variable.
Before running the program, do this in the terminal in this directory first.


python -3.10 -m venv venv

venv/Scripts/activate

python -m pip install --upgrade pip

pip uninstall -y torch torchvision torchaudio ultralytics opencv-python opencv-python-headless

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -U ultralytics opencv-python

pip install 'uvicorn[standard]'  

pip install fastapi



After installing all the python libraries, run the server.py
python server.py

Then open ninjutsu-v10.html to start the game in your browser.