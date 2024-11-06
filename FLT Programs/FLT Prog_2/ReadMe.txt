Programming assignment 2
Name:  Srimalini Sankara Narayana Nanduri
Course: Formal Language Theory CS 5313
Due Date: 09/24/2024

ReadMe:

Compile the Program:
javac FLTprog_2.java

Run the Program:
Run: java FLTprog_2 input.txt

Programs and Subprograms:

> Main Program (FLTprog_2): It handles reading the input file, parsing the DFSM, and coordinating the conversion to a regular expression.

> createRegex: This method constructs the regex table by adding a new start state and a new final state, filling in the transitions based on the input transition table.

> eliminateStates: This method eliminates intermediate states from the regex table and builds the final regular expression by processing transitions.

> removeEps: This method removes occurrences of "eps" (epsilon) from the final regular expression to ensure a simplified output.

> simplifyRegex: This method applies simplification rules to the generated regular expression to ensure it is in its simplest form.

> splitString: This method splits a string based on a specified delimiter and handles spaces appropriately.

> Error Handling: The program includes error handling to catch issues such as incorrect formatting in the input file, ensuring robust execution.

Files Submitted:

FLTprog_2.java
input.txt
ReadMe
log.txt
log1.txt
log2.txt
log3.txt
log4.txt
log5.txt