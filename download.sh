rsync -avz --exclude-from './.syncignore.txt'  -e "ssh -p 2222" --progress ktaebum@warhol1.snu.ac.kr:/home/ktaebum/Workspace/compression/* .
