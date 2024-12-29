Changes for 0.4.1:
- changed Makefile to accept external pd-lib-builder

[ipoke~] version 0.4.0 for Pd

Original MaxMsp class [ipoke~] by Pierre Alexandre Tremblay. Pd port by Katja Vetter 2012, with assistance from Matt Barber, Julian Brooks, Alexander Harker, Charles Henry and P.A. Tremblay.

Changes for 0.4.0:
- changed to the pd-lib-builder build system,
- reorganized the directory structure,
- fixed  the bool switch issue,
- added an ipoke~-meta.md file,
- uploaded code to a github repository.

Fred Jan Kraan, fjkraan@xs4all.nl, 2017-04-10


Original .3 README.txt:


********************************************************************************

INSTALL

You must have Pure Data or Pd-extended installed to use [ipoke~] for Pd. Some test patches use cyclone/poke~] (for comparisons), the rest is all vanilla Pd. 

Copy or move directory [ipoke~] somewhere on your computer. Do not alter the directory structure or directory names as the test patches use it to find the path to the executable. However you can copy the executable for your platform from 'bin' or 'bin64' to Pd's 'extra' folder to use [ipoke~] globally.

Directory 'bin' contains executables for Windows (32 bit), Linux (32 bit) and OSX (fat binary). Directory 'bin64' contains an executable for Linux 64 bit. You may need to remove the 32 bit executable if you're on Linux 64.

********************************************************************************

BUILD

If you want to build the [ipoke~] executable yourself (for Windows with MinGW, OSX or Linux), cd to directory 'src' and type 'make'. 

********************************************************************************

CONTACT

If you find a bug in this version, please mail to:

katjavetter@gmail.com

********************************************************************************

Katja Vetter, Aug. 2012
