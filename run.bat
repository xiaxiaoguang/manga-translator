@echo off
:: 切换到当前脚本所在的目录
:: 这可以防止在“以管理员身份运行”时出现找不到文件的错误
cd /d "%~dp0"

:: 检查 G 盘和 E 盘是否存在 (只是简单的检查，防止报错)
IF NOT EXIST "G:\" (
    echo 错误: 找不到 G 盘。请检查移动硬盘是否连接。
    pause
    exit
)
IF NOT EXIST "E:\" (
    echo 错误: 找不到 E 盘。请检查路径配置。
    pause
    exit
)

:: Loop from 3 to 11 (Start, Step, End)
FOR /L %%i IN (4,1,5) DO (
    echo ----------------------------------------------------------------
    echo Processing A%%i...
    echo ----------------------------------------------------------------
    
    :: Command 1: Manga Translator
    :: Using double quotes " for paths is standard on Windows to handle spaces correctly
    python -m manga_translator local --use-gpu -v -i "G:\manga\J%%i" --config-file "E:\MangaTranslator\manga-image-translator\examples\config-example.toml" --context-size 10
    
    :: Command 2: WebP to PDF
    python webptopdf.py "G:\manga\J%%i-translated" "G:\manga\translated\J%%i.pdf"

)

@REM python webptopdf.py "G:\manga\G1-translated" "G:\manga\G2-translated" "G:\manga\G3-translated" "G:\manga\G4-translated" "G:\manga\G5-translated" "G:\manga\translated\Magazine.pdf"

echo All tasks completed.
pause