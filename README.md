# The Friday Branch.

Lets see - Our goals are ....

# AND

TODO 
+ Create a Directory Tree and add as a file?
+ USB
+ RPi port (which RPi?)

# Notes
+ Beaglebone build maybe broken, updated git commands and website, using the build environment to do a sanity check.
  
# build notes - raw stuff and things
`
rpi_sdimage.sh</br>
 ├── sanity checks</br>
 │    ├── CROSS_COMPILE exists</br>
 │    ├── objcopy exists</br>
 │    └── firmware files present</br>
 ├── build boot objects</br>
 │    ├── entry.S (ARMv6)</br>
 │    ├── boot monitor</br>
 │    └── kernel</br>
 ├── link with custom linker script</br>
 │    └── output kernel.elf</br>
 ├── objcopy → kernel.img<//br>
 ├── prepare FAT filesystem</br>
 │    ├── start.elf</br>
 │    ├── bootcode.bin</br>
 │    ├── config.txt</br>
 │    └── kernel.img</br>
 └── done

`
