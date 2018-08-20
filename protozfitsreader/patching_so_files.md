# Where does this come from?

The contents of this repo come entirely from: http://www.isdc.unige.ch/~lyard/repo/

## Modifications with respect to http://www.isdc.unige.ch/~lyard/repo/

1. setup.py:

    I just updated the setup.py a little to use setuptools instead of
    distutils. The result is that all `*.py` and `*.so` files of this package are
    installed into a folder called "protozfitsreader" inside the "site-packages",
    before the files were directly copied into "site-packages" which worked as
    well, but was a little untidy.

2. relative imports

    As a result, I had to modify some `import` statement, they were written as
    absolute imports, but now we do relative imports from the "protozfitsreader"
    package.


3. set RPATH

   The main purpose of this was, to deliver "patched" so-files. The so-files
   in the original tar-ball contain absolute RPATHs, which prevent the linker from
   finding the dependencies. Here I simply patched the so-files and added relative
   RPATH like this:

    ```
    for i in *.so; do patchelf --set-rpath '$ORIGIN' $i; done
    for i in raw*.so; do patchelf --set-rpath '$ORIGIN:$ORIGIN/../../../' $i; done
    for i in *.so; do echo $i; patchelf --print-rpath $i; done
    ```

    gives:
    ```
    libACTLCore.so
    $ORIGIN
    libZFitsIO.so
    $ORIGIN
    rawzfitsreader.cpython-35m-x86_64-linux-gnu.so
    $ORIGIN:$ORIGIN/../../../
    rawzfitsreader.cpython-36m-x86_64-linux-gnu.so
    $ORIGIN:$ORIGIN/../../../
    ```
