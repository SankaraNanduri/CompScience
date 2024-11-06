public class Remove_UnitProduction {
    public static void eliminateUnitProductions(char[][][] grammar) {
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < grammar.length; i++) {
                for (int j = 0; j < grammar[i].length; j++) {
                    if (grammar[i][j] != null && grammar[i][j].length == 1) {
                        char symbol = grammar[i][j][0];
                        if (symbol >= 'A' && symbol <= 'Z') {
                            // This is a unit production
                            int ntIndex = symbol - 'A';  // Find the non-terminal on the RHS
                            // Add all productions of the RHS non-terminal to LHS
                            for (int k = 0; k < grammar[ntIndex].length; k++) {
                                if (grammar[ntIndex][k] != null && grammar[ntIndex][k].length > 0) {
                                    grammar[i][j] = grammar[ntIndex][k];  // Replace with RHS production
                                }
                            }
                            grammar[i][j] = null;  // Remove the unit production
                            changed = true;
                        }
                    }
                }
            }
        }
    }
}
