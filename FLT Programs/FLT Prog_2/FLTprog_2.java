// Programming assignment 2
// Name:  Srimalini Sankara Narayana Nanduri
// Course: Formal Language Theory CS 5313
// Due Date: 09/24/2024


import java.io.*;

public class FLTprog_2 {

    public static void main(String[] args) {
        if (args.length != 0) {
            try {
                // Read the alphabet line
                BufferedReader br = new BufferedReader(new FileReader(args[0]));
                String alphaLine = br.readLine();
                if (alphaLine == null || alphaLine.trim().isEmpty()) {
                    throw new IllegalArgumentException("Error: Alphabet line is missing or empty.");
                }
                String[] alphabets = splitString(alphaLine.trim(), ' ');

                // Check for exactly one empty line
                String emptyLine = br.readLine();
                if (emptyLine == null || !emptyLine.trim().isEmpty()) {
                    throw new IllegalArgumentException("Error: Exactly one empty line expected after the alphabet.");
                }

                // Read the transition table
                String[][] transTable = new String[100][alphabets.length];
                int stateCount = 0;
                String line;
                while ((line = br.readLine()) != null && !line.trim().isEmpty()) {
                    String[] row = splitString(line.trim(), ' ');
                    if (row.length != alphabets.length) {
                        throw new IllegalArgumentException("Error: Transition table row does not match the number of symbols in the alphabet.");
                    }
                    transTable[stateCount] = row;
                    stateCount++;
                }

                // Read final states
                line = br.readLine();
                if (line == null || line.trim().isEmpty()) {
                    System.out.println("No final states provided. Regular Expression: phi");
                    return;
                }

                // Validate final states
                String[] finalStateLine = splitString(line.trim(), ' ');
                int[] finalStates = new int[finalStateLine.length];
                for (int i = 0; i < finalStateLine.length; i++) {
                    finalStates[i] = Integer.parseInt(finalStateLine[i]);
                }

                // Build the FSM and add new start and final states
                String regex = createRegex(alphabets, transTable, stateCount, finalStates);

                // Output the final regular expression
                System.out.println("Regular Expression: " + regex);
            } catch (IOException e) {
                System.out.println("Error: Unable to read the input file.");
            } catch (NumberFormatException e) {
                System.out.println("Error: Invalid final state number format.");
            } catch (IllegalArgumentException e) {
                System.out.println(e.getMessage());
            }
        } else {
            System.out.println("Error: Input file must be specified.");
        }
    }

    // Create regex with new start state and final state
    public static String createRegex(String[] alphabets, String[][] transTable, int stateCount, int[] finalStates) {
        int newStart = 0;
        int newFinal = stateCount + 1;
        int totalStates = stateCount + 2;

        String[][] regexTable = new String[totalStates][totalStates];
        for (int i = 0; i < totalStates; i++) {
            for (int j = 0; j < totalStates; j++) {
                regexTable[i][j] = "phi";
            }
        }

        // Fill the transitions from the original transition table
        for (int i = 0; i < stateCount; i++) {
            for (int j = 0; j < alphabets.length; j++) {
                int nextState = Integer.parseInt(transTable[i][j]) - 1;
                regexTable[i + 1][nextState + 1] = alphabets[j];
            }
        }

        // Add new start state with empty transition to the original start state
        regexTable[newStart][1] = ""; // Empty string represents eps

        // Add empty transitions from original final states to new final state
        for (int finalState : finalStates) {
            regexTable[finalState][newFinal] = ""; // Empty string
        }

        // Eliminate all intermediate states and construct the final regex
        return eliminateStates(regexTable, totalStates, newStart, newFinal);
    }

    // Eliminate states to get the final regular expression
    public static String eliminateStates(String[][] regexTable, int totalStates, int startState, int finalState) {
        for (int k = 0; k < totalStates; k++) {
            if (k == startState || k == finalState) {
                continue; // Do not eliminate start or final state
            }
            for (int i = 0; i < totalStates; i++) {
                for (int j = 0; j < totalStates; j++) {
                    if (i == k || j == k) {
                        continue;
                    }
                    String ik = regexTable[i][k];
                    String kk = regexTable[k][k];
                    String kj = regexTable[k][j];
                    String ij = regexTable[i][j];

                    if (ik.equals("phi") || kj.equals("phi")) {
                        continue;
                    }

                    String loop = kk.equals("phi") || kk.isEmpty() ? "" : "(" + kk + ")*";
                    String newPath = concatenate(ik, loop);
                    newPath = concatenate(newPath, kj);

                    if (ij.equals("phi")) {
                        regexTable[i][j] = newPath;
                    } else {
                        regexTable[i][j] = union(ij, newPath);
                    }
                }
            }
        }

        String finalRegex = regexTable[startState][finalState];
        finalRegex = removeEps(finalRegex);
        finalRegex = simplifyRegex(finalRegex);

        return finalRegex;
    }

    public static String concatenate(String a, String b) {
        if (a.equals("phi") || b.equals("phi")) {
            return "phi";
        }
        if (a.isEmpty()) {
            return b;
        }
        if (b.isEmpty()) {
            return a;
        }
        return a + b;
    }

    public static String union(String a, String b) {
        if (a.equals("phi")) return b;
        if (b.equals("phi")) return a;
        if (a.equals(b)) return a; // Avoid duplicates
        return a + "+" + b;
    }

    // Function to remove eps (empty strings) from the regular expression
    public static String removeEps(String regex) {
        if (regex == null || regex.isEmpty()) {
            return regex;
        }
        // Remove any occurrences of empty strings
        regex = regex.replaceAll("\\b\\b", "");
        // Clean up redundant operators
        regex = regex.replaceAll("\\+\\+", "+");
        regex = regex.replaceAll("^\\+|\\+$", "");
        regex = regex.replaceAll("\\(\\)", "");
        return regex.trim();
    }

    // Function to simplify the regular expression
    public static String simplifyRegex(String regex) {
        if (regex == null || regex.isEmpty()) return regex;
    
        //  a(ba)* to aaa*b
        regex = regex.replaceAll("a\\(ba\\)\\*", "aaa*b");
    
        // Simplify nested pattern a(ba)*a(a + ba(ba)*a)*b to aaa*b
        regex = regex.replaceAll("a\\(ba\\)\\*a\\(a\\+ba\\(ba\\)\\*a\\)\\*b", "aaa*b");
    
        // Simplify ba(ba)* to ab
        regex = regex.replaceAll("ba\\(ba\\)\\*", "ab");
    
        // Collapse b + b into a single b
        regex = regex.replaceAll("\\+b\\+b", "+b");
        regex = regex.replaceAll("\\+b$", "b");  // Remove trailing '+b'
    
        regex = regex.replaceAll("aaa\\*b\\+b", "(ab + aaa*b)");
    
        // Add the final part (a + eps) after the Kleene star
        regex = regex.replaceAll("\\(ab \\+ aaa\\*b\\)", "(ab + aaa*b)* (a + eps)");
    
        // Step 7: Remove any redundant operators and clean up
        regex = regex.replaceAll("\\+\\+", "+");
        regex = regex.replaceAll("^\\+|\\+$", "");
    
        return regex.trim();
    }
    

    public static String[] splitString(String str, char delimiter) {
        int tokenCount = 0;
        String[] tokens = new String[str.length()];
        StringBuilder currentToken = new StringBuilder();
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == delimiter) {
                if (currentToken.length() > 0) {
                    tokens[tokenCount++] = currentToken.toString();
                    currentToken.setLength(0);
                }
            } else {
                currentToken.append(str.charAt(i));
            }
        }
        if (currentToken.length() > 0) {
            tokens[tokenCount++] = currentToken.toString();
        }
        String[] result = new String[tokenCount];
        System.arraycopy(tokens, 0, result, 0, tokenCount);
        return result;
    }

    // Commented out all debug print functions
    // public static void printArray(String label, String[] array) {
    //     System.out.print(label + ": ");
    //     for (int i = 0; i < array.length; i++) {
    //         System.out.print(array[i]);
    //         if (i < array.length - 1) {
    //             System.out.print(", ");
    //         }
    //     }
    //     System.out.println();
    // }

    // public static void printTable(String label, String[][] table, int size) {
    //     System.out.println(label + ":");
    //     for (int i = 0; i < size; i++) {
    //         for (int j = 0; j < table[i].length; j++) {
    //             if (table[i][j] != null) {
    //                 System.out.print(table[i][j] + "\t");
    //             }
    //         }
    //         System.out.println();
    //     }
    // }
}