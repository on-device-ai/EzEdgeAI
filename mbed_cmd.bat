@REM *** McuML Project ***
@REM Copyright (C) 2020 On-Device AI Co., Ltd.
@CALL conda activate mbed
@IF %ERRORLEVEL% EQU 0 goto Procedures
@CALL conda env create --file mbed.yml --name mbed
@CALL conda activate mbed
:Procedures
%*
:End
@CALL conda deactivate

