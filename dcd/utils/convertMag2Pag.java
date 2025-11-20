// package utils;

// danish mod: have commented below 3 lines since they were generic and instead imported specific classes (the 3 lines below that)
// import edu.cmu.tetrad.data.*;
// import edu.cmu.tetrad.graph.*;
// import edu.cmu.tetrad.search.*;
import edu.cmu.tetrad.graph.Graph;
import edu.cmu.tetrad.graph.GraphSaveLoadUtils;
// danish mod: commented out below line
// import edu.cmu.tetrad.search.utils.MagToPag;
import edu.cmu.tetrad.graph.GraphTransforms;
import java.io.File;

public class convertMag2Pag {

    private static void convert(String path){
        // danish mod: changed variable name graph to magPath for clarity
        File magPath = new File(path);

        if (!magPath.getName().endsWith(".mag")){
            System.out.println(magPath.getName().concat(" --- DOES NOT EXIST"));
            return;
        }

        // danish mod: added print statements in the try block for debugging
        try {
            // danish mod: the loadGraphTxt function is no longer in GraphUtils, it is in GraphSaveLoadUtils
            // System.out.println("DANISH: the path is " + path);
            Graph mag = GraphSaveLoadUtils.loadGraphTxt(magPath);
            // System.out.println("DANISH: This is the mag input graph as a string:\n " + mag);
            // System.out.println("DANISH: loaded MAG");
            // danish mod: commented out below 2 lines since I am directly using GraphTransforms
            // MagToPag obj = new MagToPag(mag);
            // obj.setCompleteRuleSetUsed(true);
            // danish mod: convert now needs an argument which decides whether to check if the MAG is legal
            // danish mod: commented out below line since directly using GraphTransforms
            // Graph pag = obj.convert(true);
            // System.out.println("DANISH: converted MAG to PAG");
            // System.out.println("DANISH: this is the converted pag graph (via search util) as a string:\n " + pag);
            // danish used other method to convert to pag (which calls the above function internally)
            Graph new_pag = GraphTransforms.magToPag(mag);
            // System.out.println("\nDANISH: converted new method MAG to PAG");
            // System.out.println("DANISH: this is the converted pag graph (via graph transforms) as a string:\n " + new_pag);
            File out = new File(magPath.getParent().concat("/").concat(magPath.getName()).concat(".pag"));
            // danish mod: the saveGraph function is no longer in GraphUtils, it is in the GraghSaveLoadUtils
            GraphSaveLoadUtils.saveGraph(new_pag, out, false);
            // System.out.println("DANISH: written pag to file");
        }

        catch(Exception e){
            System.out.println(magPath.getName().concat(" --- ERROR"));
        }
    }

    public static void main(String[] args) {
        convert(args[0]);
    }
}
