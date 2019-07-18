---
title: How to Replace Computer by Beach Time - The Magic of the Shell
header:
  overlay_image: /images/introduction-to-bash-overlay.jpg
  overlay_filter: 0.5 # same as adding an opacity of 0.5 to a black background
  caption: "Photo by [**Athena Lam on Unsplash**](https://unsplash.com/@thecupandtheroad)"
  actions:
    - label: "Enjoying the Beach in San Sebastian"
classes: wide

tags:
  - bash
excerpt: A comprehensive introduction to shell scripting.
---

The more I program, the lazier I become. I just can't see, why I should be doing something that a computer can do so much better, faster and more reliably on its own. On my way to the lazy worker, I have found shell scripting to be a great friend and helper. Whilst in the past, you could often get away using Windows (if you wanted), in a time, where more and more computing is outsourced to the cloud and servers are mostly Linux based, in my mind every data scientist and data engineer should have at least a basic understanding of shell scripting.
(Bonus: \*See the appendix at the bottom for my favourite time, when shell scripting actually made me happy. Hint: It did involve plenty of beach time)
Since I believe that deep down we all want to spend our time doing other things than manually moving files around, I thought I would share with you an introduction to my personal essentials of the shell. Hopefully, it can help make your life a bit easier (and spend more time at the beach if you wish to do so), too :). 

## Basics: Moving Around and Basic File Manipulations
### Moving Around
If you log on, the first thing you probably want to know is where you are (hint: You will probably be in your home directory). You can find this out by printing your current working directory to the screen:
```
# print working directory
pwd
```

Next, you should list its contents by typing
```
# list contents of current directory
ls
```

Many bash commands allow for modifiers, so-called flags. They mostly consist of a single letter, which is appended to the command by a "-". You can combine multiple flags by writing them one behind the other. `ls` has a multitude of possible flags. Here are some examples

```
# include hidden files
ls -a

# include hidden files and print more details 
ls -la

# list only directory (without content)
ls -1d directoryname
```

In order to find out more about a command use the manual (man pages): 
```
# print manual for mycommand
man mycommand

# for example:
man ls
```

In order to move around, you use the `cd` (change directory) command:
```
# change to directory called mydirectory inside current directory
cd mydirectory

# change to directory above
cd ..

# move to directory which is also inside the directory above (basically a "parallel" directory)
cd ../mydirectory

# change into previous work directory
cd -
```

### Advanced Moving Around
You can use `pushd`/`popd` to add/delete directories from/to the stack. Once added to the stack, you can jump between the directories in the stack. Note that when building your stack, you need to add the final directory twice, since the final position will always get overwritten (it sounds more complicated than it is, just try it out and you will see what I mean).
```
# add mydirectory to stack
pushd mydirectory

# show directories in stack
dirs -v 

# delete top repository from stack
popd 

# change to directory numbered n (eg 2) in the stack
cd ~2
```

### Basic Interaction with Files and Folders
You can create a simple text file by 
```
# create a text file called mynewtextfile.txt
touch mynewtextfile.txt
```

Files are copied, moved or deleted by:
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

Directories are copied, moved and deleted like files. However, copying and deleting requires the `-r` (recursive) flag:
```
# copy directory
cp -r folder_old folder_new

# delete directory
rm –r folder_to_remove

# rename directory (does not require -r flag)
mv old_folder new_folder
```

## Interacting with Files and Chaining Commands Together - Slightly Less Basic
### Interacting with Text Files
Now that we know how to move files around, we also want to do something useful with them. 

There are four main options to access the contents of a text file. I recommend just trying them out to see what they do and how they behave differently. 
```
# prints whole file to screen
cat mytextfile

# prints file to screen one screenful at a time
more mytextfile

# prints file to screen, allowing for backwards movement and returns to previous screen view after finishing
# note: less does not require the whole text to be read and therefore will start faster on large text files then more or text-editors
less mytextfile

# use a text editor (for example nano or here vi)
vi mytextfile
```
About the choice of editor: Personally, I am a big fan of Vim. However, I do admit that it does have a bit of a steep learning curve at first. If you feel like starting out with sth a bit more beginner-friendly, you could take a look at nano. However, keep VIM in mind for the future, the speed-up for textprocessing is amazing once you know your way around. 

You can also return the first or last n rows of a document
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
grep -in mybadfilename *.py
```

In the last example, we have seen an example of a place holder. \*.py denotes all files with a .py ending. 

### Redirecting Output
Some commands print to the screen. To re-direct the output to a file we can use `>` and `>>`. `>>` appends the output to an existing file or creates a new file if the file does not exist yet. In contrast, `>` always creates a new file. If a file with the same name already exists, it overwrites it. Here is an example of how to re-direct the output of the `grep -in mybadfilename *.py` command to a file:
```
# creates new file; if file exists, overwrites it
mycommand > mytextfile
# example:
grep -in mybadfilename *.py > myoutputfile

# appends output to file; if myoutputfile does not exist yet, creates it
mycommand >> mytextfile
# exammple:
grep -in mybadfilename *.py >> myoutputfile
```
If in addition to re-directing the output to the file, we **also** want to print the output to the screen, we can use `| tee`. Note, that the complete command needs to appear before the `|`. 
```
# print output to screen plus re-direct it to file
mycommand | tee myoutputfile

# example:
grep -in mybadfilename *.py | tee myoutputfile
```

In the previous example, we have seen the usage of the pipe (|) command. How does it work?
`|` re-directs output into functions which normally take their input "from the right", so expect the input to come after the function call. An example: As demonstrated previously, `grep` requires the syntax `grep sth filename`. However you might have a programm returning output and want to grep for something in this output. This is where the `|` comes into play. For example, `ps aux` shows all processes running on your system. You might want to search for a process containing a certain string, e.g. launch\_. This is how you do it: 
```
# grep for the string launch_ in the output of ps aux
ps aux | grep launch_
```

## Variables and Scripting
### Variables
Bash is a scripting language and not typed. Variables are defined and assigned using the `=` sign. There must not be any whitespace between the variable name, the = sign and the value. You can access the content of a variable using `$` followed by the variablename. You can use `echo` to print to the screen. 

```
# define string variable
my_string_variable="this_is_a_string"

# define numeric variable
my_numeric_variable=3

# print variable to screen
# (will print this_is_a_string to the screen)
echo $my_string_variable
```

Variables are often used to define paths and filenames. When variables are re-solved within text, it is required to put `{}` around the variable names. As an example consider the just created variable my_string_variable. Assume you want to print 'this_is_a_string_1'. In order to print the content of the variable my_string_variable, followed by \_1, use {} around the variable name:
```
# incorrect (bash will think that the variable is called "my_string_variable_1"):
echo $my_string_variable_1

# instead use:
echo ${my_string_variable}_1
```
In the second example, bash resolves the reference to this\_is\_a\_string and then appends a \_1 to the resulting string.

### Loops
Bash uses the `for ... do ... done` syntax for looping. The example shows how to use a loop to rename the files myfilename1 and myfilename2 to myfilename1.bac and myfilename2.bac. Note that there is no comma separating the elements of a list.
```
# rename files by appending a .bac to every filename
# no comma between list elements!
for myfilename in myfilename1 myfilename2
do
  mv $filename ${filename}.bac;
done
```
In order to loop over a list of integers, use the sequence generator to generate a list first:
```
for i in $(seq 1 3)
do
 echo $i
done
```
Note: `$()` opens a sub-shell, where the content of () is resolved. The results are then returned to the outer shell. In the example above `seq 1 3` produces the sequence 1 2 3 which is passed back to the outer shell, where it is then looped over. This behaviour can be used to for example loop over files containing a certain pattern:
```
for myfile in $(ls *somepattern*)
do
  cp myfile myfile.bac
done
```

### Writing and Executing a (Very) Basic Script
To create a script, create a text file containing bash syntax, make it executable and run it. Let's look at a very basic (and admittedly very useless) example. Create a file containing the following content: 
```
#!/bin/bash

# print myfilename.txt
echo "Hello World!"

exit 0
```
and save it as print\_hello\_world.sh. Note the first line of the file which tells the shell which interpreter to use. 
You make it executable by adding execution rights for the owner and run it by ./scriptname:
``` 
# add execution rights for file myfirstbashscript.sh for the owner of the file
chmod u+x print_hello_world.sh

# run 
./print_hello_world.sh
```

If instead of hard-coding "Hello World!", you want the user to pass the to-be-greeted to the script, you can pass this as a variable to the script. Let's create a new file print_hello_user.sh with the following content:
```
#!/bin/bash

# print "Hello " + user-input 
echo "Hello " $1

exit 0
```
If we give it execution rights and execute it like this
```
./print_hello_user.sh "Universe"
```
it will print "Hello Universe" to the screen. Why? "Universe" as the first input variable after the filename gets passed to the script as a variable with name 1, which is then referred to through the $1 command in the print statement. 

## Final Tips and Tricks
* Use tab-completion whenever possible: To autocomplete, press the "Tab" key. If there are multiple options, press "Tab" twice to display all options.
* `ESC + .` will bring back the last token from the previous line. Example: `cp file_a file_b`; then in the next line `ESC + .` will produce file_b.
* brace-completion: You can use `{}` to shorten your code. For example if you want to rename a file, you can type `mv myfilename{,.bac}`. This executes as `mv myfilename myfilename.bac`. Very useful for interactive work (I would not use it in scripts though).
* `tail -f myfilename`: `tail filename` produces the tail at the point of execution. However, you might want to be able to follow output scripts while they are being written. `tail -f` starts of as normal `tail`, but then keeps on appending when new lines appear at the end of the output  file.
* `watch -n somenumber command` executes the command every somenumber seconds. For example, `watch -n 2 ls` runs ls every 2 seconds. Great to watch files being transferred.

## Conclusion
In this post, we have looked at a basic introduction to using the shell. We have seen how to orient yourself in a shell environment, how to move around and some basic interactions with files. Finally, we have created and ran our first script and looked at some of my favourite tricks. While this should give you a good start, this was only a small introduction into the weird and wonderful world of shell scripting. If you are curious to learn more, [here](https://devhints.io/bash) is a good and extensive cheat-sheet for scripting, which might help you further. For a complete coverage of the topic check out Mendel Cooper's [An in-depth exploration of the art of shell scripting](http://tldp.org/LDP/abs/html/index.html). As always, [StackOverflow](https://stackoverflow.com/) also has plenty of advice and help to offer :) Have fun! 

## \*Appendix: How Shell Scripting Actually Allowed Me to Spend More Time on the Beach
I did my PhD in San Sebastian, the capital of the Basque Country and home of the famous "La Concha" beach. My thesis was very computationally focused and required orchestrating lots of different technologies. I still fondly remember setting up my computer to automatically generate a huge amount of input files for calculations, submit the calculations to the supercomputing center, wait for them to complete, extract the relevant data from the output, visualize the results, create a whole hierarchy of webpages and push all of this to a web server, so the results could be viewed by multiple people collaborating from all over the world. It did all of this fully automatically on the push of a button and did so reliably without ever making a mistake. And I? I was enjoying my lunch at the beach :) 
