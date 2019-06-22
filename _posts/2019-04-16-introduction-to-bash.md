---
title: How to Find the Time to Spend More Time on the Beach - An Introduction to the Magic of the Shell
tags:
  - bash
excerpt: A comprehensive introduction to shell scripting.
---

The more I program, the more lazy I become. I just can't see, why I should be doing sth, that a computer can do so much better, faster and more reliable on its own. On my way to the lazy worker, I have found shell scripting to be a great friend and helper. When I was doing my PhD in computational physics, I had set up a whole calculation machinery which would compile a huge amount of input files for calculations, send of all the calculations to the parallel computing service queuing system, wait for them to finish, extract the relevant data from the output, visualize the results into graphics and animations, create a whole hierarchy of webpages and push all of this onto a webserver, so the results could be viewed from multiple people collaborating from all over the world. It did all of this fully automatically on the push of a button, while I was out enjoying my lunch-break at the beach (at that time I lived in San Sebastian, home of the famous "La Concha"-beach, voted [Europe's most beautiful beach](https://www.tripadvisor.co.uk/TravelersChoice-Beaches-cTop-g4) in 2018) or for the longer calculations soundly asleep in bed. The best: It did so reliably and never made any mistakes. Because, deep down in our hearts, most of us are lazy, here is a small introduction to the power of bash programming. Hopefully, it can help to send you on your way to becoming a lazier person (and spend more time at the beach), too :) (or possibly just have more time for other interesting things in life/science/work/you-name-it).

## Basics: Moving Around and Basic File Manipulations
### Moving Around
If you log on, the first thing you probably want to know is where you are (you will probably be in your homedirectory). You can find this out by printing the name of your current working directory onto the screen
```
# print working directory
pwd
```

You can find out what is in this directory by listing its contents. Many bash commands allow for modifiers to be added, so-called flags. They most commonly consist of a single letter, which is appened to the command by a "-". You can combine multiple flags by just writing them one behind the other. For example the command `ls` lists the contents of a directory. It has a multitude of possible flags. Here are some examples

Print content directories:
```
# list current directory
ls 

# include hidden files
ls -a

# include hidden files and print more details 
ls -la

# list only directory (without content)
ls -1d directoryname
```

You can find info on a bash command and its modifiers by using the `man` command.
```
# print manual for mycommand
man mycommand

# for example:
man ls
```

In order to move around, you you the `cd` (change directory) command:

Change directory to directory newdirectory inside the current directory: 
```
# change to directory called mydirectory inside current directory
cd mydirectory

# change to directory above
cd ..

# chaining these together to move into directory which is also inside the directory above (basically into a "parallel" directory)
cd ../mydirectory

# change into previous work directory
cd -
```

### Advanced Moving Around
You can use pushd/popd to add/delete directories from/to a stack. Once added to the stack, you can jump between the directories. Note that when building up the stack, you need to add the final directory twice, since the final position will always get overwritten (it sounds more complicated than it is, just try it out and you will see what I mean).
```
# add mydirectory to stack
pushd mydirectory

# delete upmost repository from stack
popd 

# show directories in stack
dirs -v 

# change to directory numbered n (eg 2) in the stack
cd ~2
```

### Basic Interaction with Files and Folders
You can create a simple text file by 
```
# create a text file called mynewtextfile.txt
touch mynewtextfile.txt
```

This file can then be copied, moved or deleted:
```
# copy file
cp oldfilename newfilename

# move/rename file
mv oldfilename newfilename

# delete file
rm oldfilename
```

In order to create (make) a new directory:
```
mkdir mynewdirectory
```

Directories are copied, moved and deleted like files. However, copying and deleting requires the -r (recursive) flag:
```
# copy directory
cp -r folder_old folder_new

# delete directory
rm –r folder_to_remove

# rename directory (without -r flag)
mv old_folder new_folder
```

## Interacting with Files and Chaining Commands Together - Slightly Less Basic
### Interacting with Text Files
Now that we know how to move files around, we also want to do sth useful with them. 

There are four main options to access the content of a text file. I recommend just trying them out to see what they do and how they behave differently. These are the commands:
```
# print whole text file to screen
cat mytextfile

# print text file to screen one screenful at a time
more mytextfile

# print text file allowing for backwards movement returning to previous screen view after finishing
# note: less does not require the whole text to be read and therefore will start faster on large text files then more or text-editors
less mytextfile

# use a text editor (for example vi)
vi mytextfile
```

You can also show only the first or last n rows of a document
```
# show first 10 rows of a document
head -10 mytextfile

# show last 10 rows of a document
tail -10 mytextfile
```

In order to find pieces of text in a document use `grep`
```
# look for the string python in mytextfile
grep python mytextfile

# search case-insensitive
grep -i python mytextfile

# return line-numbers with the results
grep -n python mytextfile

# search for a filename ("mybadfilename" in the example) (case insensitive) in all files with the ending *.py and return the occurences together with the line number
grep -in mybadfilename *.sas
```

In the last example, we have seen an example of a place holder. \*.sas denotes all files with a .sas ending. 

### Redirecting Output
Some commands print an output to the screen. We might want to re-direct this output to a file. This can be done using `>` and `>>`. `>` creates a new file, `>>` appends to an existing file (or creates a new file if the file does not exist). For example we might want to re-direct the output of the `grep -in mybadfilename *.sas` command into a file:
```
# creates new file; if file exists, overwrites it
mycommand > mytextfile
# example:
grep -in mybadfilename *.sas > myoutputfile

# appends to file; if myoutputfile does not exists, it creates it
mycommand >> mytextfile
# exammple:
grep -in mybadfilename *.sas >> myoutputfile
```
If in addition to re-directing the output to the file, we also want to have the output on the screen, we can use `| tee`. Note, that the complete command needs to appear before the `|`. 
```
# print output to screen plus re-direct to file
mycommand | tee myoutputfile

# example:
grep -in mybadfilename *.sas | tee myoutputfile
```

`|` let's you re-direct output into functions which expect their input to come after the function call. An example: Calling `grep` on a filename requires the syntax `grep sth filename`. But you might have a programm returning output and want to grep for sth in this output. You can do this by using the `|`. For example, `ps aux` shows all processes running on your system. You might want to search for a process containing a certain string, e.g. launch_. 
```
# grep for the string launch_ in the output of ps aux
ps aux | grep launch_
```

## Variables and Scripting
Bash uses variables. You define variables by using the = sign. There must not be any whitespace between the variable name, the = sign and the value. You can access the content of a variable using `$` followed by the variablename. You can use `echo` to print to the screen. 

### Variables
```
# define variable
mynewvariable="this_is_a_string"

# print variable to screen
# (will print this_is_a_string to the screen)
echo $mynewvariable
```

Variables are often used to define paths and filenames. When variables are re-solved within text, it is required to put {} around the variable names.
```
In order to print the content of the variable mynewvariable, followed by _1, use {} around the variable name:
# incorrect
echo $mynewvariable_1

# correct
echo ${mynewvariable}_1
```

### Loops
Bash uses the for ... do ... done syntax for looping:
```
# loop over the filenames myfilename1 and myfilename2 and rename them to myfilename1.bac and myfilename2.bac
for myfilename in myfilename1, myfilename2;
do
  mv $filename ${filename}.bac;
done
```
If you want to loop over an integer, you can use the sequence generator: 
```
for i in $(seq 1 3);
do
 echo $i
done
```
Note that $() opens a new sub-shell, where it resolves the content of () and then passes the output back to the outer shell (here: seq 1 3 produces the sequence 1 2 3 and passes it back to the outer shell, where it is then looped over. This can be very useful to loop for example over files containing a certain pattern:
```
for myfile in $(ls *somepattern*);
do
  cp myfile myfile.bac
done
```

### Writing a (Very) Basic Script
You can create a text file containing bash syntax, make it executable and then run it. For example you can create a text file containing the following code snippet:
```
#!/usr/bin/bash

# print myfilename.txt
echo myfilename.txt

exit 0
```
and save it as myfirstbashscript.sh . (This file does not do anything interesting, it just prints myfilename.txt to the screen). You then need to make it executable by typing 
``` 
# add execution rights for file myfirstbashscript.sh for the owner of the file
chmod u+x myfirstbashscript.sh
```
You can then run it by typing
```
# run script myfirstbashscript.sh
./myfirstbashscript.sh
```

If instead of hard-coding myfilename.txt, you want to be able to pass this as a variable to the script, you can write it like this:
```
#!/usr/bin/bash

# print user-input to the screen
echo $1

exit 0
```
`$1` refers to the first input variable after the filename. So if you now run the script and pass a word to it, it prints the word on the screen.
```
# run script myfirstbashscript.sh with input "hello world"
./myfirstbashscript.sh "hello world"
```

## Miscellaneous Useful Stuff
* Use tab-completion whenever possible: To autocomplete, press the "Tab" key. If there are multiple options, press it twice to display all options.
* `ESC + .` will bring back the last token from the previous line (eg: `cp filea fileb`; then in the next line `ESC + .` will produce fileb)
* brace-completion: You can use `{}` to shorten your code sometimes. For example if you want to rename a file, you can type `mv myfilename{,.bac}`. This does the same as `mv myfilename myfilename.bac`.
* `tail -f myfilename`: `tail filename` produces the tail at the point of execution. However, you might want to be able to follow output scripts while they are being written. `tail -f` starts of as normal `tail`, but then keeps on appending when new lines appear at the end of the output  file.
* `watch -n somenumber command` executes the command every somenumber seconds. Great in order to watch files being transfered.

## Conclusion
This was only a small introduction into the weird and wonderful world of shell scripting. If you found this interesting and are curious to try out more, [here](https://devhints.io/bash) is a good and extensive cheat-sheet for scripting, which might help you get going. For a complete coverage of the topic check out Mendel Cooper's [An in-depth exploration of the art of shell scripting](http://tldp.org/LDP/abs/html/index.html As always, [stackoverflow](https://stackoverflow.com/) also has plenty of advice and help to offer. Or ask me :)
