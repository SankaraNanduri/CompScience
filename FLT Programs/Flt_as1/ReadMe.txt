Programming assignment 1a
Name:  Srimalini Sankara Narayana Nanduri
Course: Formal Language Theory CS 5313
Due Date: 09/02/2024

How to Run the Program:

Compile the Program:
javac FLTprog_1.java

Run the Program:
java FLTprog_1 dfsm.txt input.txt

<dfsm_file>: The name of the file containing the DFSM specification.
<input_file>: The name of the file containing the input string to be processed by the DFSM.

Subprograms(6):
main(String[] args):
The entry point of the program. It checks for the correct number of arguments, parses the DFSM file, reads the input string, and calls the simulateDFSM method.

parseDFSMFile(String fileName):
Parses the DFSM specification file to create a DFSM object. It handles the reading of the alphabet, transition table, and accepting states. Checks the file format and ensures the correct separation of sections.

readInputString(String fileName):
Reads the input string from a file. Trims any leading or trailing spaces and checks for an empty input.

simulateDFSM(DFSM dfsm, String inputString):
Simulates the DFSM using the parsed DFSM object and the input string. Processes the input string character by character, transitioning between states according to the DFSM's transition table. Checks if the final state is one of the accepting states and returns the result.

splitString(String str):
A method to split a string by spaces into an array of substrings. Similar to String.split() but implemented manually to adhere to the exercise constraints.

DFSM Class:
Represents the DFSM with its components:
alphabet: An array of characters representing the alphabet.
transitionTable: A 2D array representing the transition table.
acceptingStates: An array representing the accepting states.

Example Output:

If the DFSM accepts the input string:
DFSM accepts the string: Yes
If the DFSM does not accept the input string:
DFSM accepts the string: No