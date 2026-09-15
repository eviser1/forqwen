<#
.SYNOPSIS
    Отменяет автозапуск голосового ассистента Jarvis.

.DESCRIPTION
    Удаляет задачу "JarvisAssistant" из Планировщика заданий Windows,
    зарегистрированную скриптом autostart.ps1. Не завершает уже запущенный
    процесс ассистента — для этого используйте пункт "Выключить" в меню
    трей-иконки или Stop-ScheduledTask / диспетчер задач.
#>

$ErrorActionPreference = "Stop"

$TaskName = "JarvisAssistant"

$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Задача '$TaskName' удалена. Автозапуск отключён."
} else {
    Write-Host "Задача '$TaskName' не найдена — автозапуск уже отключён."
}
