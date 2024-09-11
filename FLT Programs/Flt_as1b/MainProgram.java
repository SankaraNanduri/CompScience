// Programming assignment 1b
// Name:  Srimalini Sankara Narayana Nanduri
// Course: Formal Language Theory CS 5313
// Due Date: 09/10/2024

import java.io.*;

public class MainProgram {

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Usage: java MainProgram <patternFile> <inputFile>");
            return;
        }

        try {
            // Step 1: Read pattern from the pattern file (input1.txt)
            String patternFile = args[0];
            String inputFile = args[1];
            String pattern = readPatternFromFile(patternFile);
            if (pattern == null || pattern.trim().isEmpty()) {
                throw new IOException("Error: The pattern file is empty or starts with an empty line: " + patternFile);
            }

            // Step 2: Build NDFSM from the pattern using NDFSMBuilder
            String ndfsmFile = "ndfsm.txt";  
            NDFSMBuilder.buildNDFSM(patternFile, ndfsmFile);  // Automatically writes to ndfsm.txt

            // Step 3: Convert NDFSM to DFSM using NDFSMtoDFSM
            String dfsmFile = "dfsm.txt";  
            NDFSMtoDFSM.main(new String[]{ndfsmFile, dfsmFile});  // Passes ndfsm.txt and dfsm.txt

            // Step 4: Test the final DFSM using FLTprog_1 (the interpreter)
            String input = readInputFromFile(inputFile);
            if (input == null || input.trim().isEmpty()) {
                throw new IOException("Error: The input file is empty: " + inputFile);
            }
            FLTprog_1.main(new String[]{dfsmFile, inputFile});
        } catch (IOException e) {
            // Print only the error message
            System.out.println(e.getMessage());
        }
    }

    // Function to read the pattern from input1.txt
    private static String readPatternFromFile(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        String pattern = reader.readLine();
        reader.close();
        return pattern;
    }

    // Function to read the input string from input.txt
    private static String readInputFromFile(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        String inputString = reader.readLine();
        reader.close();
        return inputString;
    }
}
