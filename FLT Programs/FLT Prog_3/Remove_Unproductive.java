public class Remove_Unproductive {
    public static void eliminateUnproductiveSymbols(char[][][] grammar) {
        boolean[] productive = new boolean[100];  // Track which non-terminals are productive
        boolean changed = true;

        while (changed) {
            changed = false;
            for (int i = 0; i < grammar.length; i++) {
                for (int j = 0; j < grammar[i].length; j++) {
                    if (grammar[i][j] != null && grammar[i][j].length > 0) {
                        boolean allProductive = true;
                        for (char symbol : grammar[i][j]) {
                            if (symbol >= 'A' && symbol <= 'Z' && !productive[symbol - 'A']) {
                                allProductive = false;
                                break;
                            }
                        }
                        if (allProductive && !productive[i]) {
                            productive[i] = true;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Remove unproductive non-terminals
        for (int i = 0; i < grammar.length; i++) {
            if (!productive[i]) {
                grammar[i] = new char[10][50]; // Clear all productions for unproductive non-terminals
            }
        }
    }
}
