mkdir -p build
gcc c-extensions/cube.c -o build/cube.so -Wall -O3 -fPIC -shared
