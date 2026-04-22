# MARS-SC User Installation Guide

This guide explains how a user can install and run MARS-SC from the downloaded
repository files. It also explains how to create a Windows executable with
PyInstaller.

The instructions assume you are using Windows and PyCharm.

## What You Need First

Install these before starting:

1. Python 64-bit.
2. PyCharm.
3. The downloaded MARS-SC repository files.
4. ANSYS/DPF support available on the machine if you will read ANSYS `.rst`
   result files.

## Step 1: Open PyCharm

1. Start PyCharm.
2. Choose `File > New Project`.
3. Create an empty project folder, for example:

   ```text
   C:\Users\<your-user-name>\PycharmProjects\MARS-SC
   ```

4. Do not create any Python files manually. The project will use the files from
   the downloaded repository.

## Step 2: Copy the Downloaded Files Into the Project

1. Open the folder where you downloaded or extracted the repository.
2. Copy all repository files and folders into the empty PyCharm project folder.
3. The PyCharm project root should now contain files like:

   ```text
   mars_sc_entry.py
   MARS-SC.spec
   requirements.txt
   pyinstaller_runtime_hook.py
   src
   tests
   ```

4. If PyCharm does not show the copied files immediately, right-click the
   project folder and select `Reload from Disk`.

## Step 3: Create a Virtual Environment

1. In PyCharm, open:

   ```text
   File > Settings > Project > Python Interpreter
   ```

2. Click `Add Interpreter`.
3. Select `Virtualenv Environment`.
4. Choose your installed Python interpreter.
5. Set the virtual environment location to:

   ```text
   .venv
   ```

6. Click `OK` or `Apply`.

PyCharm should now use the new `.venv` environment for this project.

## Step 4: Install Required Packages

1. Open the PyCharm terminal.
2. Make sure the terminal starts in the project root.
3. If the virtual environment is not active, run:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Upgrade pip:

   ```powershell
   python -m pip install --upgrade pip
   ```

5. Install the project requirements:

   ```powershell
   python -m pip install -r requirements.txt
   ```

Wait until installation finishes without errors.

If PowerShell does not allow the virtual environment to activate, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then close and reopen the PyCharm terminal and activate `.venv` again.

## Step 5: Run MARS-SC From PyCharm

To run the program from the terminal:

```powershell
python mars_sc_entry.py
```

You can also create a PyCharm Run button:

1. Open `Run > Edit Configurations`.
2. Click `+`.
3. Choose `Python`.
4. Set `Script path` to:

   ```text
   mars_sc_entry.py
   ```

5. Set `Working directory` to the project root.
6. Make sure the interpreter is the project `.venv`.
7. Click `Apply`, then `Run`.

## Step 6: Build a Windows Executable With PyInstaller

This step is for users who want to create a standalone executable folder.

PyInstaller does not create a traditional Windows setup wizard in this project.
It creates a runnable application folder under `dist`.

1. Open the PyCharm terminal.
2. Activate the virtual environment if needed:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install PyInstaller:

   ```powershell
   python -m pip install pyinstaller
   ```

4. Build the application:

   ```powershell
   pyinstaller MARS-SC.spec --clean
   ```

5. After the build finishes, open:

   ```text
   dist\MARS-SC
   ```

6. Run:

   ```text
   MARS-SC.exe
   ```

## Step 7: Give the Built Program to Another User

If you built the executable with PyInstaller:

1. Go to:

   ```text
   dist\MARS-SC
   ```

2. Send the whole `MARS-SC` folder to the user.
3. The user should run:

   ```text
   MARS-SC.exe
   ```

Do not send only the `.exe` file. The other files in the folder are required.

## Troubleshooting

If package installation fails:

1. Confirm the correct virtual environment is active.
2. Upgrade pip:

   ```powershell
   python -m pip install --upgrade pip
   ```

3. Try installing requirements again:

   ```powershell
   python -m pip install -r requirements.txt
   ```

If the program does not open:

1. Run it from the terminal so you can see the error:

   ```powershell
   python mars_sc_entry.py
   ```

2. Confirm these files exist in the project root:

   ```text
   mars_sc_entry.py
   requirements.txt
   src
   ```

If the PyInstaller build fails:

1. Delete old build folders:

   ```powershell
   Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
   ```

2. Run the build again:

   ```powershell
   pyinstaller MARS-SC.spec --clean
   ```

If ANSYS result files do not load, confirm that `ansys-dpf-core` installed
successfully and that the `.rst` files are accessible from the machine.
