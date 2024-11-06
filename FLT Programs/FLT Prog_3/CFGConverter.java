import java.io.*;

public class CFGConverter {
    public static String[][] grammarTable = new String[100][10];
    public static int rowCount = 0;

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.out.println("Usage: java CFGConverter <input_file>");
            return;
        }

        String inputFile = args[0];
        grammarTable = readGrammarToTable(inputFile);

        // Call transformations in sequence
        Remove_Unreachable.applyTransformation();
        Remove_Unproductive.applyTransformation();
        Remove_Epsilon.applyTransformation();
        Remove_UnitProduction.applyTransformation();
    }

    // Read grammar from file into a 2D array
    private static String[][] readGrammarToTable(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        String line;

        while ((line = reader.readLine()) != null) {
            String[] parts = line.split("::=");
            if (parts.length < 2) continue;

            String lhs = parts[0].trim();
            String[] rhs = parts[1].trim().split("\\|");

            grammarTable[rowCount][0] = lhs;
            System.arraycopy(rhs, 0, grammarTable[rowCount], 1, rhs.length);
            rowCount++;
        }
        reader.close();
        return grammarTable;
    }

    // Utility to print the grammar in BNF format
    public static void printGrammar() {
        for (int i = 0; i < rowCount; i++) {
            System.out.print(grammarTable[i][0] + " ::= ");
            for (int j = 1; j < grammarTable[i].length && grammarTable[i][j] != null; j++) {
                if (j > 1) System.out.print(" | ");
                System.out.print(grammarTable[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }
}
?????????????????????????????????????????????????????????????
import java.io.*;

public class CFGConverter {
    public static String[][] grammarTable = new String[100][10];
    public static int rowCount = 0;
    private static String originalStartSymbol = null;
    private static String newStartSymbol = "<S'>";

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.out.println("Usage: java CFGConverter <input_file>");
            return;
        }

        String inputFile = args[0];
        grammarTable = readGrammarToTable(inputFile);

        // Step 1: Add a new start symbol if necessary
        addNewStartSymbol();

        // Step 2: Apply transformations in sequence
        Remove_Unreachable.applyTransformation();
        Remove_Unproductive.applyTransformation();
	Remove_Epsilon.applyTransformation();
        Remove_UnitProduction.applyTransformation();

        // Print the final transformed grammar
	System.out.println("Final grammar:");
        printGrammar();
    }

    // Read grammar from file into a 2D array
    private static String[][] readGrammarToTable(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        String line;

        while ((line = reader.readLine()) != null) {
            String[] parts = line.split("::=");
            if (parts.length < 2) continue;

            String lhs = parts[0].trim();
            if (originalStartSymbol == null) originalStartSymbol = lhs;  // Set the original start symbol

            String[] rhs = parts[1].trim().split("\\|");

            grammarTable[rowCount][0] = lhs;
            System.arraycopy(rhs, 0, grammarTable[rowCount], 1, rhs.length);
            rowCount++;
        }
        reader.close();
        return grammarTable;
    }

    // Add a new start symbol if the original start symbol appears on RHS
    private static void addNewStartSymbol() {
        // Check if the original start symbol appears on the RHS of any production
        boolean isStartSymbolOnRHS = false;
        for (int i = 0; i < rowCount; i++) {
            for (int j = 1; j < grammarTable[i].length && grammarTable[i][j] != null; j++) {
                if (grammarTable[i][j].contains(originalStartSymbol)) {
                    isStartSymbolOnRHS = true;
                    break;
                }
            }
            if (isStartSymbolOnRHS) break;
        }

        // If the original start symbol appears on RHS, add the new start symbol production
        if (isStartSymbolOnRHS) {
            System.out.println("Adding new start symbol: " + newStartSymbol + " ::= " + originalStartSymbol);

        // Shift all rows down by one to make space for <S'>
        for (int i = rowCount; i > 0; i--) {
            grammarTable[i] = grammarTable[i - 1];
        }

        // Insert <S'> as the first row
        grammarTable[0] = new String[10];
        grammarTable[0][0] = newStartSymbol;
        grammarTable[0][1] = originalStartSymbol;
        
        rowCount++;
        markAsReachable(newStartSymbol);  // Ensure <S'> is marked as reachable
        }
    }

    // Utility function to mark the new start symbol as reachable
    private static void markAsReachable(String symbol) {
        // This function should interact with your unreachable symbols module to ensure <S'> is recognized.
        // For instance, set a flag or adjust logic so <S'> is always considered reachable.
        //System.out.println("Marking " + symbol + " as reachable.");
    }

    // Utility to print the grammar in BNF format
    public static void printGrammar() {
        for (int i = 0; i < rowCount; i++) {
            System.out.print(grammarTable[i][0] + " ::= ");
            for (int j = 1; j < grammarTable[i].length && grammarTable[i][j] != null; j++) {
                if (j > 1) System.out.print(" | ");
                System.out.print(grammarTable[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }
}
/////////////////////////////////////////////////////////

import java.io.*;

public class CFGConverter {
    public static String[][] grammarTable = new String[100][10];
    public static int rowCount = 0;
    private static String originalStartSymbol = null;
    private static String newStartSymbol = "<S'>";

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            System.out.println("Usage: java CFGConverter <input_file>");
            return;
        }

        String inputFile = args[0];
        grammarTable = readGrammarToTable(inputFile);

        // Step 1: Add a new start symbol if necessary
        addNewStartSymbol();

        // Step 2: Apply transformations in sequence
        Remove_Unreachable.applyTransformation();
        Remove_Unproductive.applyTransformation();
        Remove_Epsilon.applyTransformation();
        Remove_UnitProduction.applyTransformation();

        // Print the final transformed grammar
        System.out.println("Final grammar:");
        printGrammar();
    }

    // Read grammar from file into a 2D array, replacing "ε" with "EPSILON"
    private static String[][] readGrammarToTable(String fileName) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(fileName));
        String line;

        while ((line = reader.readLine()) != null) {
            String[] parts = line.split("::=");
            if (parts.length < 2) continue;

            String lhs = parts[0].trim();
            if (originalStartSymbol == null) originalStartSymbol = lhs;  // Set the original start symbol

            String[] rhs = parts[1].trim().split("\\|");

            for (int i = 0; i < rhs.length; i++) {
                // Replace "ε" with "EPSILON" in the right-hand side productions
                rhs[i] = rhs[i].replace("ε", "EPSILON").trim();
            }

            grammarTable[rowCount][0] = lhs;
            System.arraycopy(rhs, 0, grammarTable[rowCount], 1, rhs.length);
            rowCount++;
        }
        reader.close();
        return grammarTable;
    }

    // Add a new start symbol if the original start symbol appears on RHS
    private static void addNewStartSymbol() {
        boolean isStartSymbolOnRHS = false;
        for (int i = 0; i < rowCount; i++) {
            for (int j = 1; j < grammarTable[i].length && grammarTable[i][j] != null; j++) {
                if (grammarTable[i][j].contains(originalStartSymbol)) {
                    isStartSymbolOnRHS = true;
                    break;
                }
            }
            if (isStartSymbolOnRHS) break;
        }

        if (isStartSymbolOnRHS) {
            System.out.println("Adding new start symbol: " + newStartSymbol + " ::= " + originalStartSymbol);

            for (int i = rowCount; i > 0; i--) {
                grammarTable[i] = grammarTable[i - 1];
            }

            grammarTable[0] = new String[10];
            grammarTable[0][0] = newStartSymbol;
            grammarTable[0][1] = originalStartSymbol;
            
            rowCount++;
            markAsReachable(newStartSymbol);
        }
    }

    // Utility function to mark the new start symbol as reachable
    private static void markAsReachable(String symbol) {
        // Ensure <S'> is considered reachable
        // System.out.println("Marking " + symbol + " as reachable.");
    }

    // Utility to print the grammar in BNF format, replacing "EPSILON" back to "ε"
    public static void printGrammar() {
        for (int i = 0; i < rowCount; i++) {
            System.out.print(grammarTable[i][0] + " ::= ");
            for (int j = 1; j < grammarTable[i].length && grammarTable[i][j] != null; j++) {
                if (j > 1) System.out.print(" | ");
                // Replace "EPSILON" with "ε" while printing
                System.out.print(grammarTable[i][j].replace("EPSILON", "ε"));
            }
            System.out.println();
        }
        System.out.println();
    }
}
