public class Remove_LeftRecursion {
    public static void eliminateLeftRecursion(char[][][] grammar) {
        for (int i = 0; i < grammar.length; i++) {
            char[][] newProductions = new char[10][50];  // Store the new productions
            char[][] alpha = new char[10][50];  // Left recursive productions
            char[][] beta = new char[10][50];   // Non-left recursive productions
            int alphaCount = 0, betaCount = 0;

            for (int j = 0; j < grammar[i].length; j++) {
                if (grammar[i][j] != null && grammar[i][j].length > 0) {
                    char firstSymbol = grammar[i][j][0];
                    if (firstSymbol == 'A' + i) {
                        // Left recursion, separate alpha part
                        char[] newAlpha = new char[grammar[i][j].length - 1];
                        for (int k = 1; k < grammar[i][j].length; k++) {
                            newAlpha[k - 1] = grammar[i][j][k];
                        }
                        alpha[alphaCount++] = newAlpha;
                    } else {
                        // Non-left recursive, add to beta
                        beta[betaCount++] = grammar[i][j];
                    }
                }
            }

            if (alphaCount > 0) {
                // Eliminate left recursion
                char[] newNonTerminal = new char[]{(char) ('A' + i), '\''};  // New non-terminal A'
                for (int j = 0; j < betaCount; j++) {
                    // Add new production: A -> beta A'
                    char[] newProduction = new char[beta[j].length + 2];
                    System.arraycopy(beta[j], 0, newProduction, 0, beta[j].length);
                    newProduction[beta[j].length] = newNonTerminal[0];
                    newProductions[j] = newProduction;
                }

                // Add alpha -> alpha A'
                for (int j = 0; j < alphaCount; j++) {
                    char[] newAlphaProduction = new char[alpha[j].length + 2];
                    System.arraycopy(alpha[j], 0, newAlphaProduction, 0, alpha[j].length);
                    newAlphaProduction[alpha[j].length] = newNonTerminal[0];
                    newProductions[betaCount + j] = newAlphaProduction;
                }

                // A' -> epsilon (add production A' -> ε)
                char[] epsilonProduction = new char[0];
                newProductions[betaCount + alphaCount] = epsilonProduction;

                // Replace original productions with new ones
                grammar[i] = newProductions;
            }
        }
    }
}
