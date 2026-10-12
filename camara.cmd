@echo off
rem Usa el Python de la variable PYTHON si existe; si no, el "python" del sistema.
if not defined PYTHON set PYTHON=python
"%PYTHON%" "%~dp0camara.py" %*
