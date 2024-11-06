public class GNFConverter {
    public static void convertToGNF(char[][][] grammar) {
        for (int i = 0; i < grammar.length; i++) {
            for (int j = 0; j < grammar[i].length; j++) {
                if (grammar[i][j] != null && grammar[i][j].length > 0) {
                    char firstSymbol = grammar[i][j][0];
                    if (firstSymbol >= 'A' && firstSymbol <= 'Z') {
                        // Replace non-terminal as the first symbol in the production
                        int ntIndex = firstSymbol - 'A';
                        for (int k = 0; k < grammar[ntIndex].length; k++) {
                            if (grammar[ntIndex][k] != null && grammar[ntIndex][k].length > 0) {
                                char[] newProduction = new char[grammar[ntIndex][k].length + grammar[i][j].length - 1];
                                // Copy the replacement production from the non-terminal
                                System.arraycopy(grammar[ntIndex][k], 0, newProduction, 0, grammar[ntIndex][k].length);
                                // Append the rest of the original production
                                System.arraycopy(grammar[i][j], 1, newProduction, grammar[ntIndex][k].length, grammar[i][j].length - 1);
                                grammar[i][j] = newProduction;  // Replace with the new production
                            }
                        }
                    }
                }
            }
        }
    }
}
