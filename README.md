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
rpi_sdimage.sh
 ├── sanity checks
 │    ├── CROSS_COMPILE exists
 │    ├── objcopy exists
 │    └── firmware files present
 ├── build boot objects
 │    ├── entry.S (ARMv6)
 │    ├── boot monitor
 │    └── kernel
 ├── link with custom linker script
 │    └── output kernel.elf
 ├── objcopy → kernel.img
 ├── prepare FAT filesystem
 │    ├── start.elf
 │    ├── bootcode.bin
 │    ├── config.txt
 │    └── kernel.img
 └── done

`
