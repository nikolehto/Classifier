% Yksi tapa tulostaa verkon sisältöä, tällä hetkellä lähinnä 3-ulotteisia
% painovektoreita.
function Plotter(matrixToPlot1, matrixToPlot2)
    length = size(matrixToPlot1,1);

    x = 1:length;
    y = ones(1, length);

    if (nargin > 1)
        length = size(matrixToPlot2,1);
        x = [ x 1:length ];
        y = [ y zeros(1, length)];
        c = [matrixToPlot1; matrixToPlot2];
    else % nargin == 1
        c = matrixToPlot1;
    end
    
    graph_dotSize = 25;
    
    scatter(x,y,graph_dotSize,c,'filled')
end