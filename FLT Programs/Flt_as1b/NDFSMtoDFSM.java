// Programming assignment 1b
// Name:  Srimalini Sankara Narayana Nanduri
// Course: Formal Language Theory CS 5313
// Due Date: 09/10/2024

import java.io.*;

public class NDFSMtoDFSM {

    // function to split a string by spaces
    private static String[] splitString(String str, String delimiter) {
        int count = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.substring(i, i + 1).equals(delimiter)) {
                count++;
            }
        }
        String[] parts = new String[count + 1];
        int partIndex = 0;
        int start = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.substring(i, i + 1).equals(delimiter)) {
                parts[partIndex++] = str.substring(start, i);
                start = i + 1;
            }
        }
        parts[partIndex] = str.substring(start);
        return parts;
    }

    // Reading the NDFSM file (ndfsm.txt)
    public static String[][] readNDFSMFile(String inputFileName, String[] alphabet, boolean[] finalStates) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(inputFileName));

        StringBuilder fileContent = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            fileContent.append(line).append("\n");
        }
        String[] lines = fileContent.toString().split("\n");

        // Validate the file format
        validateFileFormat(lines);

        // Read the alphabet from the input file
        String alphabetLine = lines[0].trim();
        String[] alph = splitString(alphabetLine, " ");
        System.arraycopy(alph, 0, alphabet, 0, alph.length);  // Copy alphabet to external array

        // Skip the empty line after the alphabet
        int i = 2;

        // Read the transition table
        String[][] ndfsmTransitions = new String[100][alphabet.length];  // Store the transitions, max 100 states
        int numStates = 0;
        while (i < lines.length && !lines[i].trim().isEmpty()) {
            String[] transitions = splitString(lines[i].trim(), " ");
            for (int j = 0; j < transitions.length; j++) {
                ndfsmTransitions[numStates][j] = transitions[j].replace("[", "").replace("]", "");  // Clean brackets
            }
            numStates++;
            i++;
        }

        // Read the final states from the NDFSM
        String[] ndfsmFinalStates = splitString(lines[i + 1].trim(), " ");
        for (int j = 0; j < ndfsmFinalStates.length; j++) {
            int finalState = Integer.parseInt(ndfsmFinalStates[j].replace("[", "").replace("]", ""));
            finalStates[finalState - 1] = true;  // Mark the final state as true
        }

        reader.close();
        return ndfsmTransitions;  // Return the NDFSM transition table
    }

    // Validate the format of the NDFSM file
    public static void validateFileFormat(String[] lines) throws IOException {
        if (lines.length == 0) {
            throw new IOException("Error: The input NDFSM file is empty.");
        }

        // Check if alphabet line is missing
        if (lines[0].trim().isEmpty()) {
            throw new IOException("Error: Alphabet line is missing in the NDFSM file.");
        }

        // Validate that there is strictly one space between alphabets
        validateAlphabetLine(lines[0]);

        // Ensure there's an empty line after the alphabet
        if (!lines[1].trim().isEmpty()) {
            throw new IOException("Error: Blank line missing after the alphabet line in the NDFSM file.");
        }

        // Validate transition table and spacing
        int i = 2;
        for (; i < lines.length; i++) {
            if (lines[i].trim().isEmpty()) {
                break; // Found the empty line separating transition table from final states
            }

            // Validate transitions have strictly one space between them
            validateTransitionSpacing(lines[i]);
        }

        // Ensure a blank line between the transition table and final states
        if (!lines[i].trim().isEmpty()) {
            throw new IOException("Error: Blank line missing after the transition table in the NDFSM file.");
        }

        // Validate the final states
        if (i == lines.length - 1 || lines[i + 1].trim().isEmpty()) {
            throw new IOException("Error: Final states line is missing in the NDFSM file.");
        }
    }

    // Validate that the alphabet line has exactly one or more spaces between characters
    public static void validateAlphabetLine(String line) throws IOException {
        if (!line.matches("([a-zA-Z0-9] ?)+")) {
            throw new IOException("Error: Alphabets must be separated by exactly one or more spaces.");
        }
    }

    // Validate that transition table entries are separated by exactly one space
    public static void validateTransitionSpacing(String line) throws IOException {
        // Validate transitions are enclosed in []
        String[] transitions = line.trim().split(" ");
        for (String transition : transitions) {
            if (!transition.matches("\\[.*\\]")) {
                throw new IOException("Error: Transition table entries must be enclosed in [].");
            }
        }

        // Check if there are multiple spaces between transitions
        if (!line.matches("(\\[.*\\] ){0,}\\[.*\\]")) {
            throw new IOException("Error: Transition entries must be separated by exactly one space.");
        }

        // Ensure there's strictly one space between the brackets
        if (line.contains("  ")) {
            throw new IOException("Error: Transition table entries must have exactly one space between them.");
        }
    }

    // Convert NDFSM transitions to DFSM transitions
    public static Object[] convertNDFSMToDFSM(String[][] ndfsmTransitions, String[] alphabet, boolean[] finalStates) {
        String[][] dfsmTransitions = new String[100][alphabet.length];  // DFSM transition table
        int[][] stateQueue = new int[100][100];  // Queue to manage DFSM states, assume max 100 states
        int[] queueSizes = new int[100];  // Track size of each state in the queue
        int[] stateMapping = new int[100];  // Maps NDFSM state sets to unique DFSM state IDs
        boolean[] dfsmFinalStates = new boolean[100];  // Track DFSM final states
        int queueSize = 0;  // Number of DFSM states in the queue
        int nextStateID = 1; // Start assigning new states from 1

        // Start with the initial NDFSM state
        stateQueue[queueSize][0] = 1;
        queueSizes[queueSize] = 1;
        stateMapping[queueSize] = nextStateID++;
        queueSize++;

        // Process each state in the queue
        for (int queueIndex = 0; queueIndex < queueSize; queueIndex++) {
            int[] currentState = stateQueue[queueIndex];  // Get the current DFSM state from the queue
            int currentStateSize = queueSizes[queueIndex];  // Size of the current state set
            String[] currentDFSMRow = new String[alphabet.length];  // DFSM transitions for this state

            // Final States Logic
            for (int j = 0; j < currentStateSize; j++) {
                int state = currentState[j] - 1;
                if (finalStates[state]) {
                    dfsmFinalStates[queueIndex] = true;
                    break;
                }
            }

            // For each symbol in the alphabet, get the reachable states
            for (int i = 0; i < alphabet.length; i++) {
                int[] reachableStates = new int[100];  // To store reachable states for this symbol
                int reachableSize = 0;

                // For each state in the current DFSM state, get the reachable states from the NDFSM
                for (int j = 0; j < currentStateSize; j++) {
                    int state = currentState[j];
                    int stateIndex = state - 1;

                    if (ndfsmTransitions[stateIndex][i] != null && !ndfsmTransitions[stateIndex][i].equals("[]") && !ndfsmTransitions[stateIndex][i].equals("")) {
                        String[] targetStates = splitString(ndfsmTransitions[stateIndex][i], ",");
                        for (int k = 0; k < targetStates.length; k++) {
                            if (!targetStates[k].trim().isEmpty()) {
                                int targetState = Integer.parseInt(targetStates[k]);
                                if (!contains(reachableStates, reachableSize, targetState)) {
                                    reachableStates[reachableSize++] = targetState;  // Add new state
                                }
                            }
                        }
                    }
                }

                // Process reachable states
                if (reachableSize > 0) {
                    int newStateID = getStateID(reachableStates, reachableSize, stateQueue, queueSize, stateMapping, nextStateID);
                    if (newStateID == nextStateID) {  // New state found
                        stateQueue[queueSize] = reachableStates;
                        queueSizes[queueSize] = reachableSize;
                        stateMapping[queueSize] = nextStateID++;  // Assign new state ID
                        queueSize++;
                    }
                    currentDFSMRow[i] = String.valueOf(newStateID);  // Store the transition to the new state
                } else {
                    currentDFSMRow[i] = "-";  // No transition
                }
            }

            // Add the transition row to the DFSM transition table
            dfsmTransitions[queueIndex] = currentDFSMRow;
        }

        return new Object[]{dfsmTransitions, dfsmFinalStates, stateMapping, queueSize};
    }

    // Write the DFSM to a file (dfsm.txt)
    public static void writeDFSMToFile(String outputFileName, String[][] dfsmTransitions, String[] alphabet, boolean[] dfsmFinalStates, int[] stateMapping, int queueSize) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName));

        // Write the alphabet
        for (int i = 0; i < alphabet.length && alphabet[i] != null; i++) {
            writer.write(alphabet[i] + " ");
        }
        writer.newLine();
        writer.newLine();  // Add a blank line after the alphabet

        // Write the transition table
        for (int i = 0; i < queueSize; i++) {
            for (int j = 0; j < alphabet.length; j++) {
                String transition = dfsmTransitions[i][j];
                if (transition == null || transition.equals("-")) {
                    writer.write(" ");
                } else {
                    writer.write(transition + " ");
                }
            }
            writer.newLine();
        }
        writer.newLine();  // Add a blank line after the transition table

        // Write the final states
        for (int i = 0; i < queueSize; i++) {
            if (dfsmFinalStates[i]) {
                writer.write(stateMapping[i] + " ");
            }
        }
        writer.newLine();

        writer.close();
    }

    // function to get the unique ID of a state
    private static int getStateID(int[] stateSet, int size, int[][] stateQueue, int queueSize, int[] stateMapping, int nextStateID) {
        for (int i = 0; i < queueSize; i++) {
            if (queueSizesMatch(stateQueue[i], stateSet, size)) {
                return stateMapping[i]; // Return the existing state ID
            }
        }
        return nextStateID; // Return new state ID if not found
    }

    // function to check if two state sets are equal
    private static boolean queueSizesMatch(int[] queueState, int[] newState, int size) {
        for (int i = 0; i < size; i++) {
            if (queueState[i] != newState[i]) return false;
        }
        return true;
    }

    // function to check if an array contains a value
    private static boolean contains(int[] array, int size, int value) {
        for (int i = 0; i < size; i++) {
            if (array[i] == value) return true;
        }
        return false;
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.out.println("Usage: java NDFSMtoDFSM <inputFile> <outputFile>");
            return;
        }

        // alphabet and final states array
        String[] alphabet = new String[100];
        boolean[] finalStates = new boolean[100];

        // Read the NDFSM file and get the transition table
        String[][] ndfsmTransitions = readNDFSMFile(args[0], alphabet, finalStates);

        // Convert NDFSM transitions to DFSM transitions
        Object[] dfsmInfo = convertNDFSMToDFSM(ndfsmTransitions, alphabet, finalStates);
        String[][] dfsmTransitions = (String[][]) dfsmInfo[0];
        boolean[] dfsmFinalStates = (boolean[]) dfsmInfo[1];
        int[] stateMapping = (int[]) dfsmInfo[2];
        int queueSize = (int) dfsmInfo[3];

        // Write the DFSM to the output file
        writeDFSMToFile(args[1], dfsmTransitions, alphabet, dfsmFinalStates, stateMapping, queueSize);
    }
}
