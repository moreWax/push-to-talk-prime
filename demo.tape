Output "assets/vhs-polished-demo-silent.webm"
Output "assets/vhs-polished-demo.gif"

Require prime-agent
Require node
Set Shell "bash"
Set FontFamily "Menlo"
Set FontSize 16
Set Width 1280
Set Height 560
Set Padding 30
Set Margin 20
Set MarginFill "#11111b"
Set BorderRadius 14
Set WindowBar "Colorful"
Set Theme "Catppuccin Mocha"
Set Framerate 20

Hide
Type "rm -f /tmp/ptt-vhs-v2-service.json /tmp/ptt-vhs-v2-service.json.lock"
Enter
Type "PTT_ENABLED=1 PTT_PRESET=smooth PTT_DEVICE=mps PTT_CONFIG=/tmp/ptt-vhs-missing.json PTT_SERVICE_STATE=/tmp/ptt-vhs-v2-service.json PTT_DEMO_AUDIO_FILE=$PWD/assets/demo-input.wav PTT_DEMO_TIMELINE=1 prime-agent --no-session --offline -ne -e $PWD/extensions/push-to-talk.ts"
Enter
Sleep 9s
Show
Sleep 1s
Space@40ms 155
Sleep 10s
