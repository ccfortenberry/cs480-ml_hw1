# cs480-ml HW1: Decition Trees and Naive Bayes
I hope this is an antithesis as to why mathematicians should never be allowed to create programming languages. Octave is a stupid, poorly-designed mess that honestly I'm amazed anyone can get something done with it. Did I bother to learn the language enough to actually script with proper SE principles in mind? No, and after trying to do machine learning with it I don't care to. When I figured out that matrix access are the exact same as FUNCTION CALLS and the fact the environment saves all variables declared in scripts even after execution (*which are still valid for all future executions as well (may whatever god you believe in help you if you name a variable std then try to use the builtin standard deviation function)*) and even the fact that doing a matrix access on an **rvalue** is a perfectly reasonable thing to do according to Octave (ie. 0.8384(4) is legal syntax... no really), I gave up any notion that rational people made this language, thus why what I wrote is in irrationally bad form (yet it still works!). So if you ready through the code and wonder "Wow! This could all be cleaned up by using functions!" feel free to do so yourself.

## Prerequisite software
Go [here](https://www.gnu.org/software/octave/) and install the correct version for your operating system
