mkdir -p build
gcc c-extensions/cube.c -o build/cube.so -Wall -O3 -fPIC -shared
gcc c-extensions/hashmap.c c-extensions/unique.c -o build/unique.so -Wall -O3 -fPIC -shared
