Output "assets/vhs-live-demo-silent.webm"
Output "assets/vhs-live-demo.gif"

Require prime-agent
Require node

Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 18
Set Width 1100
Set Height 650
Set Padding 24
Set Margin 16
Set MarginFill "#11111b"
Set BorderRadius 12
Set WindowBar "Colorful"
Set Theme "Catppuccin Mocha"
Set Framerate 20
Set TypingSpeed 1ms

Hide
Type "rm -f /tmp/ptt-vhs-service.json /tmp/ptt-vhs-service.json.lock"
Enter
Show
Type "PTT_ENABLED=1 PTT_PRESET=balanced PTT_DEVICE=mps PTT_CONFIG=/tmp/ptt-vhs-missing.json PTT_SERVICE_STATE=/tmp/ptt-vhs-service.json PTT_DEMO_AUDIO_FILE=$PWD/assets/demo-input.wav prime-agent --no-session --offline -ne -e $PWD/extensions/push-to-talk.ts"
Enter
Sleep 9s
Space@40ms 105
Sleep 6s
Ctrl+C
Sleep 250ms
Ctrl+C
Sleep 1s
