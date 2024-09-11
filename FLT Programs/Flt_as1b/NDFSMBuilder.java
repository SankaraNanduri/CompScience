// Programming assignment 1b
// Name:  Srimalini Sankara Narayana Nanduri
// Course: Formal Language Theory CS 5313
// Due Date: 09/10/2024

import java.io.*;

public class NDFSMBuilder {

    public static void buildNDFSM(String inputFileName, String outputFileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(inputFileName));
        PrintWriter writer = new PrintWriter(new FileWriter(outputFileName));

        // Read the pattern from input file
        String pattern = reader.readLine();
        reader.close();

        // Handle case where file is empty or not properly read
        if (pattern == null || pattern.trim().isEmpty()) {
            System.err.println("Error: The input file is empty or the pattern could not be read.");
            writer.close();
            return;
        }

        pattern = pattern.trim(); // Remove leading/trailing spaces

        // Step 1: Build the alphabet (unique symbols in the pattern)
        StringBuilder alphabetBuilder = new StringBuilder();
        for (int i = 0; i < pattern.length(); i++) {
            if (alphabetBuilder.indexOf(String.valueOf(pattern.charAt(i))) == -1) {
                alphabetBuilder.append(pattern.charAt(i)).append(" ");
            }
        }
        String alphabet = alphabetBuilder.toString().trim();

        // Write the alphabet to the file
        writer.println(alphabet);
        writer.println();  // Empty line to separate sections

        // Build the transition table 
        int numStates = pattern.length() + 1;  // States are now numbered from 1 to pattern.length() + 1

        for (int state = 1; state <= numStates; state++) {
            StringBuilder transitionSet = new StringBuilder();

            for (int i = 0; i < alphabet.length(); i += 2) {  // Increment by 2 because alphabet has spaces between characters
                char currentChar = alphabet.charAt(i);

                // First state (1) transitions to state 2 on 'a'
                if (state == 1) {
                    if (currentChar == pattern.charAt(0)) {
                        transitionSet.append("[").append(state).append(",").append(2).append("] ");
                    } else {
                        transitionSet.append("[").append(state).append("] ");
                    }
                }
                // Intermediate states transition to the next state on matching symbol
                else if (state > 1 && state < numStates) {
                    if (currentChar == pattern.charAt(state - 1)) {
                        transitionSet.append("[").append(state + 1).append("] ");
                    } else {
                        transitionSet.append("[] ");
                    }
                }
                // Final state 
                else if (state == numStates) {
                    transitionSet.append("[").append(state).append("] ");
                }
            }
            writer.println(transitionSet.toString().trim());  // Trim and print the row
        }

        writer.println();  // Empty line to separate sections

        // Define the final state 
        writer.printf("[%d]\n", numStates);

        writer.close();
        //System.out.println("NDFSM built and written to " + outputFileName);
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.out.println("Usage: java NDFSMBuilder <inputFile> <outputFile>");
            return;
        }

        buildNDFSM(args[0], args[1]);
    }
}
