@ECHO OFF
@REM cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

START python server.py
@REM # Sleep for 10s to give the server enough time to start and dowload the dataset
timeout 10  

FOR /L %%i IN (0,1,9) DO (
  ECHO Starting client %%i
  START python client.py --client-id=%%i
)

ECHO All clients started!

@REM # Enable CTRL+C to stop all background processes
@REM trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
@REM # Wait for all background processes to complete
@REM wait