@echo off
set SERVER_IP=159.65.61.54
set ZIP_FILE=ctapipe_deploy.zip

REM Automatically change to the project root (parent directory of this script)
cd /d "%~dp0.."

echo Running from: %CD%
echo Packaging code for deployment...

rem Create a temporary deployment directory if it doesn't exist
if not exist "deploy" mkdir deploy

rem Copy source code
if exist "src" (
    echo Copying src...
    xcopy /E /I /Y src deploy\src
)

rem Copy documentation
if exist "docs" (
    echo Copying docs...
    xcopy /E /I /Y docs deploy\docs
)

rem Copy examples
if exist "examples" (
    echo Copying examples...
    xcopy /E /I /Y examples deploy\examples
)

rem Copy setup files
if exist "pyproject.toml" copy pyproject.toml deploy\
if exist "setup.cfg" copy setup.cfg deploy\
if exist "setup.py" copy setup.py deploy\
if exist "README.md" copy README.md deploy\
if exist "LICENSE" copy LICENSE deploy\

echo.
echo Deployment package created in the 'deploy' directory.
echo To move to Linux server, use scp or similar:
echo scp -r deploy/* user@linux-server:/path/to/ctapipe/
echo.
echo NOTE: Ensure you have installed dependencies on the Linux server:
echo pip install -e .[all]
echo.
pause
