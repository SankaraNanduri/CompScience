import java.util.HashSet;
import java.util.Set;

public class Remove_Unreachable {
    public static void applyTransformation() {
        Set<String> reachable = new HashSet<>();
        reachable.add(CFGConverter.grammarTable[0][0]);

        boolean changed;
        do {
            changed = false;
            for (int i = 0; i < CFGConverter.rowCount; i++) {
                if (reachable.contains(CFGConverter.grammarTable[i][0])) {
                    for (int j = 1; j < CFGConverter.grammarTable[i].length && CFGConverter.grammarTable[i][j] != null; j++) {
                        String[] symbols = CFGConverter.grammarTable[i][j].split(" ");
                        for (String symbol : symbols) {
                            if (symbol.startsWith("<") && symbol.endsWith(">") && reachable.add(symbol)) {
                                changed = true;
                            }
                        }
                    }
                }
            }
        } while (changed);

        int newRow = 0;
        for (int i = 0; i < CFGConverter.rowCount; i++) {
            if (reachable.contains(CFGConverter.grammarTable[i][0])) {
                CFGConverter.grammarTable[newRow++] = CFGConverter.grammarTable[i];
            }
        }
        CFGConverter.rowCount = newRow;

        System.out.println("After removing unreachable symbols:");
        CFGConverter.printGrammar();
    }
}
