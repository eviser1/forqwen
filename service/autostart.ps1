<#
.SYNOPSIS
    Регистрирует автозапуск голосового ассистента Jarvis в Планировщике заданий Windows.

.DESCRIPTION
    Создаёт задачу "JarvisAssistant", которая:
      - запускается при входе текущего пользователя в систему;
      - выполняет pythonw.exe без консольного окна (у pythonw.exe окна нет по
        определению, дополнительно сама задача помечается скрытой в Планировщике);
      - при аварийном завершении автоматически перезапускается через 1 минуту,
        не более 3 попыток подряд.

    Скрипт идемпотентен: если задача с таким именем уже существует, она сначала
    удаляется, а затем регистрируется заново с текущими параметрами. Повторный
    запуск не создаёт дублей.

.NOTES
    Запускать из обычной PowerShell-консоли (права администратора не требуются
    для задачи уровня текущего пользователя с LogonType Interactive).
#>

$ErrorActionPreference = "Stop"

$TaskName    = "JarvisAssistant"
$ProjectRoot = "D:\claude code prj\3\jarvis"
$ScriptPath  = Join-Path $ProjectRoot "orchestrator.py"

# --- Поиск pythonw.exe (запуск без консольного окна) ---------------------
$pythonwCmd = Get-Command "pythonw.exe" -ErrorAction SilentlyContinue
if ($pythonwCmd) {
    $PythonExe = $pythonwCmd.Source
} else {
    Write-Warning "pythonw.exe не найден в PATH. Убедитесь, что Python 3.11+ установлен и добавлен в PATH."
    $PythonExe = "pythonw.exe"
}

if (-not (Test-Path $ScriptPath)) {
    Write-Warning "Файл оркестратора не найден: $ScriptPath"
    Write-Warning "Задача будет зарегистрирована, но не сможет запуститься, пока файл не появится."
}

# --- Идемпотентность: удаляем существующую задачу с тем же именем --------
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Найдена существующая задача '$TaskName' — удаляю перед повторной регистрацией."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# --- Действие: pythonw.exe orchestrator.py --------------------------------
$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument ('"{0}"' -f $ScriptPath) `
    -WorkingDirectory $ProjectRoot

# --- Триггер: вход текущего пользователя в систему -------------------------
$Trigger = New-ScheduledTaskTrigger -AtLogOn

# --- Настройки: скрытая задача, без ограничения времени выполнения, ------
# --- перезапуск через 1 минуту при сбое, максимум 3 попытки --------------
$Settings = New-ScheduledTaskSettingsSet `
    -Hidden `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# --- Принципал: запуск от имени текущего пользователя, интерактивный сеанс -
$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Локальный голосовой ассистент Jarvis: автозапуск при входе в систему." `
    | Out-Null

Write-Host "Готово: задача '$TaskName' зарегистрирована."
Write-Host "Ассистент запустится автоматически при следующем входе в систему."
Write-Host "Запустить прямо сейчас, не перезаходя в систему:"
Write-Host "    Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Отключить автозапуск: service\uninstall_autostart.ps1"
