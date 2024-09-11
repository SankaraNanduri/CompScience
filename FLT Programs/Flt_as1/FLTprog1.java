import java.io.*;
import java.util.Arrays;

public class FLTprog1 {

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Usage: java FLTprog1 <dfsm_file> <input_file>");
            return;
        }

        String dfsmFileName = args[0];
        String inputFileName = args[1];

        try {
            DFSM dfsm = parseDFSMFile(dfsmFileName);

            // Check if DFSM object is null (indicating an error during parsing)
            if (dfsm == null) {
                System.err.println("Error: Failed to parse DFSM file. ");
                return;
            }

            // Check if the alphabet is present and valid
            if (dfsm.alphabet == null || dfsm.alphabet.length == 0) {
                System.err.println("Error: Alphabet is not present or is empty.");
                return;
            }

            // Check if the transition table is present and valid
            if (dfsm.transitionTable == null || dfsm.transitionTable.length == 0) {
                System.err.println("Error: Transition table is not present or is empty.");
                return;
            }
            
            // Check if there are accepting states
            if (dfsm.acceptingStates == null || dfsm.acceptingStates.length == 0) {
                System.err.println("Error: No accepting states defined.");
                return;
            }

            String inputString = readInputString(inputFileName);
            
            // Check if the input string is empty
            if (inputString == null ) { //|| inputString.isEmpty()
                //System.err.println("Error: Input string is empty.");
                return;
            }

            boolean result = simulateDFSM(dfsm, inputString);
            System.out.println(result ? "yes" : "no");

        } catch (IOException e) {
            System.err.println("Error reading files: " + e.getMessage());
        }
    }

    // Method to parse the DFSM specification file and create a DFSM object
    private static DFSM parseDFSMFile(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        char[] alphabet = null;
        int[][] transitionTable = null;
        int[] acceptingStates = null;

        String line;
        int section = 1;
        int stateIndex = 0;  // Declare stateIndex to track the current row in the transition table
        boolean lastLineWasEmpty = false;  // Track if the last line was empty
        int emptyLineCount = 0;

        // Check if the file is empty
        if ((line = reader.readLine()) == null) {
            System.err.println("Error: DFSM file is empty.");
            reader.close();
            return null;
        }

        do {
            line = line.trim();

            if (line.isEmpty()) {
                emptyLineCount++;
                if (emptyLineCount > 1) {
                    System.err.println("Error: Only one empty line is permitted between sections.");
                    return null;
                }
                lastLineWasEmpty = true;
                section++;
                continue;
            }

            if (lastLineWasEmpty) {
                emptyLineCount = 0;
                lastLineWasEmpty = false;
            }

            if (section == 1) {
                // Alphabet section
                //System.out.println("Parsing alphabet section...");
                String[] symbols = line.split("\\s+");  // Split on one or more spaces
                if (symbols.length == 0) {  // Ensure there is at least 1 symbol
                    System.err.println("Error: Alphabet must have at least one symbol.");
                    return null;
                }
                alphabet = new char[symbols.length];
                for (int i = 0; i < symbols.length; i++) {
                    alphabet[i] = symbols[i].charAt(0);
                }
                //System.out.println("Alphabet: " + Arrays.toString(alphabet));

            } else if (section == 2) {
                // Transition Table section
                //System.out.println("Parsing transition table section...");
                String[] parts = splitString(line);
                if (transitionTable == null) {
                    transitionTable = new int[100][parts.length]; // Initial guess for number of states, will resize later
                }
                if (stateIndex >= transitionTable.length) {
                    // If we've reached the current capacity, double the size of the transition table
                    transitionTable = Arrays.copyOf(transitionTable, transitionTable.length * 2);
                }
                transitionTable[stateIndex] = new int[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    transitionTable[stateIndex][i] = Integer.parseInt(parts[i]);
                }
                stateIndex++;
                //System.out.println("Transition Table Row: " + Arrays.toString(transitionTable[stateIndex - 1]));

            } else if (section == 3) {
                // Accepting States section
                //System.out.println("Parsing accepting states section...");
                String[] states = line.split("\\s+");  // Split on one or more spaces for multiple states
                acceptingStates = new int[states.length];
                for (int i = 0; i < states.length; i++) {
                    acceptingStates[i] = Integer.parseInt(states[i]);
                }
                //System.out.println("Accepting States: " + Arrays.toString(acceptingStates));
            }
        } while ((line = reader.readLine()) != null);

        reader.close();
        // Resize the transition table to the actual number of states parsed
        if (transitionTable != null) {
            transitionTable = Arrays.copyOf(transitionTable, stateIndex);
        }
        return new DFSM(alphabet, transitionTable, acceptingStates);
    }

    // Method to read the input string from a file
    private static String readInputString(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        String inputString = reader.readLine();
    
        // Check if the file is empty (inputString is null)
        if (inputString == null) {
            System.err.println("Error: Input file is empty.");
            reader.close();
            return null;
        }
    
        inputString = inputString.trim();
    
        // Check if the trimmed input string is empty
        if (inputString.isEmpty()) {
            System.err.println("Error: Input string is empty.");
            reader.close();
            return null;
        }
    
        reader.close();
        return inputString;
    }

    // Method to simulate the DFSM with the given input string
    private static boolean simulateDFSM(DFSM dfsm, String inputString) {
        int currentState = 1; // Start state is always state 1
        int alphabetLength = dfsm.alphabet.length;

        // Create a map for alphabet indexing
        int[] alphabetIndex = new int[128]; // ASCII size, initialized to -1
        for (int i = 0; i < alphabetIndex.length; i++) {
            alphabetIndex[i] = -1;
        }
        for (int i = 0; i < alphabetLength; i++) {
            alphabetIndex[dfsm.alphabet[i]] = i;
        }

        // Process each character in the input string
        for (int i = 0; i < inputString.length(); i++) {
            char c = inputString.charAt(i);
            int index = alphabetIndex[c];
            if (index == -1) {
                System.err.println("Error: Invalid character '" + c + "' in input string.");
                return false; // Invalid character in input string
            }
            
            // Check if currentState is within the bounds of the transition table
            if (currentState - 1 >= dfsm.transitionTable.length || currentState - 1 < 0) {
                System.err.println("Error: Current state " + currentState + " is out of bounds in the transition table.");
                return false;
            }
            
            // Check if the transition index is within the bounds of the transition table row
            if (index >= dfsm.transitionTable[currentState - 1].length) {
                System.err.println("Error: Transition index " + index + " is out of bounds for the current state.");
                return false;
            }

            currentState = dfsm.transitionTable[currentState - 1][index];
        }

        // Check if the final state is an accepting state
        boolean isAccepted = false;
        for (int i = 0; i < dfsm.acceptingStates.length; i++) {
            if (dfsm.acceptingStates[i] == currentState) {
                isAccepted = true;
                break;
            }
        }
        return isAccepted;
    }

    // Primitive method to split a string by spaces (like String.split())
    private static String[] splitString(String str) {
        int count = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ') {
                count++;
            }
        }
        String[] parts = new String[count + 1];
        int partIndex = 0;
        int start = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ') {
                parts[partIndex++] = str.substring(start, i);
                start = i + 1;
            }
        }
        parts[partIndex] = str.substring(start);
        return parts;
    }
}

// Class to represent the DFSM with its components: alphabet, transition table, and accepting states
class DFSM {
    char[] alphabet;         // Array of symbols in the alphabet
    int[][] transitionTable; // Transition table (2D array)
    int[] acceptingStates;   // Array of accepting states

    public DFSM(char[] alphabet, int[][] transitionTable, int[] acceptingStates) {
        this.alphabet = alphabet;
        this.transitionTable = transitionTable;
        this.acceptingStates = acceptingStates;
    }
}