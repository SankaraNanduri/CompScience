public class Remove_Epsilon {
    public static void applyTransformation() {
        //System.out.println("Starting Epsilon Removal");

        // Print the initial grammar table to verify the content
        //System.out.println("Initial Grammar Table:");
        for (int i = 0; i < CFGConverter.rowCount; i++) {
            //System.out.print(CFGConverter.grammarTable[i][0] + " ::= ");
            for (int j = 1; j < CFGConverter.grammarTable[i].length && CFGConverter.grammarTable[i][j] != null; j++) {
                //System.out.print(CFGConverter.grammarTable[i][j] + " | ");
            }
            //System.out.println();
        }

        // Step 1: Identify nullable non-terminals and directly remove EPSILON
        String[] nullable = new String[100];
        int nullableCount = 0;

        for (int i = 0; i < CFGConverter.rowCount; i++) {
            for (int j = 1; j < CFGConverter.grammarTable[i].length && CFGConverter.grammarTable[i][j] != null; j++) {
                // Trim whitespace around each symbol to ensure matching
                CFGConverter.grammarTable[i][j] = CFGConverter.grammarTable[i][j].trim();
                
                if ("EPSILON".equals(CFGConverter.grammarTable[i][j])) {
                    // Detect EPSILON and add to nullable
                    //System.out.println("Detected EPSILON in production: " + CFGConverter.grammarTable[i][0] + " ::= EPSILON");
                    nullable[nullableCount++] = CFGConverter.grammarTable[i][0];
                    CFGConverter.grammarTable[i][j] = null; // Remove EPSILON
                    //System.out.println("Marked " + CFGConverter.grammarTable[i][0] + " as nullable and removed EPSILON.");
                }
            }
        }
        //System.out.println("Final Nullable Non-Terminals:");
        for (int i = 0; i < nullableCount; i++) {
            //System.out.println(nullable[i]);
        }

        // Step 2: Generate alternative productions for nullable symbols
        for (int i = 0; i < CFGConverter.rowCount; i++) {
            for (int j = 1; j < CFGConverter.grammarTable[i].length && CFGConverter.grammarTable[i][j] != null; j++) {
                String[] symbols = CFGConverter.grammarTable[i][j].split(" ");
                for (String nullableSymbol : nullable) {
                    if (nullableSymbol == null) break;
                    if (containsSymbol(symbols, nullableSymbol)) {
                        String alternative = generateProductionWithoutSymbol(symbols, nullableSymbol);
                        addProductionToGrammar(CFGConverter.grammarTable, CFGConverter.rowCount, CFGConverter.grammarTable[i][0], alternative);
                        //System.out.println("Added alternative production for " + CFGConverter.grammarTable[i][0] + ": " + alternative);
                    }
                }
            }
        }

        // Step 3: Final cleanup and output
        int newRow = 0;
        for (int i = 0; i < CFGConverter.rowCount; i++) {
            String lhs = CFGConverter.grammarTable[i][0];
            String[] nonNullProductions = new String[CFGConverter.grammarTable[i].length];
            int newProdIndex = 1;
	    boolean hasProduction = false;

            for (int j = 1; j < CFGConverter.grammarTable[i].length && CFGConverter.grammarTable[i][j] != null; j++) {
                if (!"EPSILON".equals(CFGConverter.grammarTable[i][j])) {
                    nonNullProductions[newProdIndex++] = CFGConverter.grammarTable[i][j];
		    hasProduction = true;
                }
            }
	    if (hasProduction) { // Only add if there's at least one valid production
            nonNullProductions[0] = lhs;
            CFGConverter.grammarTable[newRow++] = nonNullProductions;
        }
	}
        CFGConverter.rowCount = newRow;

        // Print final grammar directly
        System.out.println("EPSILON productions:");
        for (int i = 0; i < CFGConverter.rowCount; i++) {
            System.out.print(CFGConverter.grammarTable[i][0] + " ::= ");
            for (int j = 1; j < CFGConverter.grammarTable[i].length && CFGConverter.grammarTable[i][j] != null; j++) {
                if (j > 1) System.out.print(" | ");
                System.out.print(CFGConverter.grammarTable[i][j]);
            }
            System.out.println();
        }
    }

    // Helper to check if symbols contain a nullable symbol
    private static boolean containsSymbol(String[] symbols, String nullableSymbol) {
        for (String symbol : symbols) {
            if (symbol.equals(nullableSymbol)) return true;
        }
        return false;
    }

    // Helper to create production without specific nullable symbol
    private static String generateProductionWithoutSymbol(String[] symbols, String nullableSymbol) {
        StringBuilder result = new StringBuilder();
        for (String symbol : symbols) {
            if (!symbol.equals(nullableSymbol)) {
                if (result.length() > 0) result.append(" ");
                result.append(symbol);
            }
        }
        return result.toString();
    }

    // Add a new production if not already present
    private static void addProductionToGrammar(String[][] grammarTable, int rowCount, String lhs, String newProduction) {
        for (int i = 0; i < rowCount; i++) {
            if (grammarTable[i][0].equals(lhs)) {
                for (int j = 1; j < grammarTable[i].length; j++) {
                    if (grammarTable[i][j] == null) {
                        grammarTable[i][j] = newProduction;
                        return;
                    }
                    if (grammarTable[i][j].equals(newProduction)) return;  // Avoid duplicates
                }
            }
        }
    }
}
