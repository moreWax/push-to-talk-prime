Output "assets/vhs-live-demo-silent.webm"
Output "assets/vhs-live-demo.gif"

Require prime-agent
Require node
Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 20
Set Width 1100
Set Height 480
Set Padding 28
Set Margin 18
Set MarginFill "#11111b"
Set BorderRadius 14
Set WindowBar "Colorful"
Set Theme "Catppuccin Mocha"
Set Framerate 20

Hide
Type "rm -f /tmp/ptt-vhs-v2-service.json /tmp/ptt-vhs-v2-service.json.lock"
Enter
Type "PTT_ENABLED=1 PTT_PRESET=balanced PTT_DEVICE=mps PTT_CONFIG=/tmp/ptt-vhs-missing.json PTT_SERVICE_STATE=/tmp/ptt-vhs-v2-service.json PTT_DEMO_AUDIO_FILE=$PWD/assets/demo-input.wav prime-agent --no-session --offline -ne -e $PWD/extensions/push-to-talk.ts"
Enter
Sleep 9s
Show
Sleep 1s
Space@40ms 175
Sleep 6s
