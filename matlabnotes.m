
%-------------------------------------------------------------------------
% Alem Fitwi (PhD), 
%-------------------------------------------------------------------------
% MATLAB Cheatsheets for Image Processing
% Windows: CMD Window, Current Folder, Workspace, & History window.
%-------------------------------------------------------------------------
% A. IMAGE PROCESSING and PLOTS
%-------------------------------------------------------------------------
% Read Image File
impath = 'C:\Users\afitwi1\Documents\MATLAB\lena.png';
img = imread(impath);
%-------------------------------------------------------------------------
% Split an RGB Image into color channels

% Read in original RGB image.
rgbImage = imread(impath);

% Extract color channels.
redChannel = rgbImage(:,:,1); % Red channel
greenChannel = rgbImage(:,:,2); % Green channel
blueChannel = rgbImage(:,:,3); % Blue channel

% Create an all black channel.
allBlack = zeros(size(rgbImage, 1), size(rgbImage, 2), 'uint8');

% Create color versions of the individual color channels.
just_red = cat(3, redChannel, allBlack, allBlack);
just_green = cat(3, allBlack, greenChannel, allBlack);
just_blue = cat(3, allBlack, allBlack, blueChannel);

% Recombine the individual color channels to create the original RGB image 
recombinedRGBImage = cat(3, redChannel, greenChannel, blueChannel);

% Display them all.
subplot(3, 3, 2);
imshow(rgbImage);
fontSize = 20;
title('Original RGB Image', 'FontSize', fontSize)
subplot(3, 3, 4);
imshow(just_red);
title('Red Channel in Red', 'FontSize', fontSize)
subplot(3, 3, 5);
imshow(just_green)
title('Green Channel in Green', 'FontSize', fontSize)
subplot(3, 3, 6);
imshow(just_blue);
title('Blue Channel in Blue', 'FontSize', fontSize)
subplot(3, 3, 8);
imshow(recombinedRGBImage);
title('Recombined to Form Original RGB Image Again', 'FontSize', fontSize)

% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
% set(gcf, 'Toolbar', 'none', 'Menu', 'none');
% Give a name to the title bar.
set(gcf, 'Name', 'Demo by ImageAnalyst', 'NumberTitle', 'Off')
%-------------------------------------------------------------------------
%Write Image
imwrite(img,'lena2.jpg');
%-------------------------------------------------------------------------
%Show Image
imshow(img);
% display multiple image in same window
%
subplot(1,2,1),imshow(img);
title('First Image');
subplot(1,2,2),imshow(img);
title('Second Image');
% using figure  
figure;
imshow(img);
%-------------------------------------------------------------------------
% Dimensions of Image
% get the dimensions of image
% if img is colour image then dimension is vector with 3 elements
% if img is grayscale image then dimension is vector wit 2 dimensions
%
 dimensions = size(img);
% number of rows
 nr = dimensions(1);
 % number of columns
 nc = dimensions(2);
 % number of color channels
 nc = dimensions(2);
%-------------------------------------------------------------------------
% Image Type Conversion (Color Scale Conversion)

% Convert RGB image to colormap to gray scale image 
imgray = rgb2gray(img);

% Convert grayscale or binary image to indexed image
imind = gray2ind(imgray); 

% Convert matrix to grayscale image
A = [1:100; 201:300]
imgray = mat2gray(A); % where A is a 2D matrix

 %%
 % Split multichannel image into its individual channels
 % eg. if you read an rgb image it has 3 color channel Red, Green & Blue
 % if one needs to fetch individual channel from that you can use the 
 % following funtion
 %
 [ r, g, b] = imsplit(img); 
 % r contains red channel of image; g contains green channnel of image; 
 % b contains blue channel of image;  
%-------------------------------------------------------------------------
% Graphs & Plots

% histogram plot of the gray scale image file
imhist(imgray);
%-------------------------------------------------------------------------
% bar chart
 X = [100,200,300,400,500,600,700]
 Y = [3,6,9,12,15,18,21]
  
 % plot X vs Y plot
 bar(X,Y);
  
% plot X
bar(X)
%-------------------------------------------------------------------------
% line plot
X = linspace(-20*pi,20*pi,100);
Y1 = sin(X);
Y2 = cos(X)
plot(X,Y1);       % plots X vs sin(X) plot
subplot(1,3,3),plot(X,Y1,X,Y2);  % plots X vs sin(X) AND X vs cos(X) plot
%-------------------------------------------------------------------------
% Customize your plot
% give title of the plot
bar(X,Y);
title('My Data Plot');
%-------------------------------------------------------------------------
% change the color of the plots
  % ------------------------------------------------------------------
  % | Short Name | Color Name | RGB Triplet | Hexadecimal Color Code |
  % ------------------------------------------------------------------
  % | 'r'        | 'red'      | [1 0 0]     | '#FF0000'              |
  % | 'g'        | 'green'    | [0 1 0]     | '#00FF00'              |
  % | 'b'        | 'blue'     | [0 0 1]     | '#0000FF'              |
  % | 'c'        | 'cyan'     | [0 1 1]     | '#00FFFF'              |
  % | 'm'        | 'magenta'  | [1 0 1]     | '#FF00FF'              |
  % | 'k'        | 'black'    | [1 1 0]     | '#FFFF00'              |
  % | 'w'        | 'white'    | [0 0 0]     | '#000000'              |
  % | 'y'        | 'yellow'   | [1 1 1]     | '#FFFFFF'              |
  % ------------------------------------------------------------------
%-------------------------------------------------------------------------
bar(X,'r'); % this plots red coloured bar graph
  
% plot X vs sin(X) AND X vs cos(X) plots
X = linspace(-2*pi,2*pi,100);
Y1 = sin(X);
Y2 = cos(X);
plot(X,Y1,X,Y2);  
  
% give label to the plot
xlabel('-2\pi < X < 2\pi');
ylabel('Sine and Cosine Values');
  
% add legend to the plot
legend({'y = sin(x)','y = cos(x)'},'Location','northeast');
%-------------------------------------------------------------------------
% change line style of the line graph
  % -------------------------------------------
  % |  Value |           Description          |
  % -------------------------------------------
  % | '--'   | Dashed line                    |
  % | '-'    | Solid line (default)           |
  % | ':'    | Dotted line                    |
  % | '-.'   | Dash-dot line                  |
  % -------------------------------------------
  plot(X,Y1,'--');  % change line style to dashed line
  
  % change line markers of the line graph
  % -------------------------------------------
  % |  Value |           Description          |
  % -------------------------------------------
  % | 'o'    | Circle                         |
  % | '+'    | Plus sign                      |
  % | 'x'    | Cross                          |
  % | '*'    | Asterisk                       |
  % | '.'    | Dot                            |
  % | 's'    | Square                         |
  % | 'd'    | Diamond                        |
  % | 'p'    | Pentagram (Five pointed star)  |
  % | 'h'    | Hexagram (Six pointed star)    |
  % | '>'    | Right pointing triangle        |
  % | '<'    | Left pointing triangle         |
  % | '^'    | Upward pointing triangle       |
  % | 'v'    | Downward pointing triangle     |
  % -------------------------------------------
  % this shows the line in the form of sequence of small circles 
  plot(X,Y1,'o');   
  % combine multiple properties in one
  % shows dashed line with circle in red for sine plot and dashed line 
  % with cross in blue for cosine plot
  plot(X,Y1,'--or',X,Y2,'--xb');

%-------------------------------------------------------------------------
img = imread('lena.png'); % Read image
red = img(:,:,1); % Red channel
green = img(:,:,2); % Green channel
blue = img(:,:,3); % Blue channel
a = zeros(size(img, 1), size(img, 2));
just_red = cat(3, red, a, a);
just_green = cat(3, a, green, a);
just_blue = cat(3, a, a, blue);
back_to_original_img = cat(3, red, green, blue);
figure, imshow(img), title('Original image')
figure, imshow(just_red), title('Red channel')
figure, imshow(just_green), title('Green channel')
figure, imshow(just_blue), title('Blue channel')
figure, imshow(back_to_original_img), title('Back to original image')
%-------------------------------------------------------------------------
% B. FUNCTION, LOOPING STRUCTURES, and OPERATORS
%-------------------------------------------------------------------------
% Factorial
%function fact = (n)
%function f = fact(n)
      %f =1; 
      %for i = 1:n       
          %f = f * i;
      %end
%end
%-------------------------------------------------------------------------
%For Loop
  % for loop from 1 to n
  n = 10;
  for i = 1:n
      fprintf("i=%d ",i)      
  end      
%-------------------------------------------------------------------------
%While Loop  
c =0;
while c<3;
    fprintf("i= %d", c)
    c=c+1;
end
%-------------------------------------------------------------------------
%If Condition
c = 1;
if mod(c,2)== 0
    fprintf("even")
elseif mod(c,2)==1
    fprintf("odd")
else
    fprintf("Neither") 
end
%-------------------------------------------------------------------------
%Arithmatic Operations
  
% Addition  
a = 3 + 5;

% Subtraction
a = 3 - 5;

% Multiplication
a = 3 * 5;

% Division
a = 3 / 5;

% Power
a = 3 ^ 5;
% OR
a = power(3,5);
  
% Logarithm
% Natural Log
a = log(3);

% Common Log
a = log10(3);

% Square Root
a = sqrt(9);

%-------------------------------------------------------------------------
% Bitwise Logical Operations
% Logical - OR
 a = bitor(3,5);
 
 % Logical - AND
 a = bitand(3,5);
 
% Logical - NOT
%a = bitnot(3,5);

% Logical - XOR
a = bitxor(3,5);
%-------------------------------------------------------------------------
%Matrix Operations

% Declare Column Vector
mat = [1; 2; 3];

% Declare Row Vector
mat = [1 2 3];
  
% OR
mat = [1, 2, 3];
  
% Declare Matrix of m x n dimension
  %
  % 1 2 3
  % 4 5 6
  % 7 8 9
  %
  
 mat = [1 2 3; 4 5 6; 7 8 9];
  
 % OR
 mat = [1, 2, 3; 4, 5, 6; 7, 8, 9];
  
% Fetch Elements From Matrix
% Fetch single element
 a = mat(1,1);
 % OR
 a = mat(1);    % numerical indexing starting from 1 in column major order
 % Fetch whole column
 a = mat(:,1);
 % Fetch whole row
 a = mat(1,:);
 % Fetch perticular part from matrix
 a = mat(1:2,2:3);  
 %%
 % Find Dimension of Matrix
 %  
 dimension = size(mat);
 
 % Declare Matrix with 1
  % 
  % 1 0 0
  % 0 1 0
  % 0 0 1
  %  
  i_mat = eye(3);
  % OR
  i_mat = eye(3,3);
  
  % Declare Matrix with 1
  % 
  % 1 1 1
  % 1 1 1
  % 1 1 1
  %
  
  i_mat = ones(3);
  % OR
  i_mat = ones(3,3);
  
  %%
  % Declare Matrix with 0
  % 
  % 0 0 0
  % 0 0 0
  % 0 0 0
  %
  
  z_mat = zeros(3);
  % OR
  z_mat = zeros(3,3);
  
  %%
  % Declare Magic Matrix
  %
  % 8 1 6 
  % 3 5 7
  % 4 9 2
  %  
  magic_mat = magic(3);
  % Addition Operation on Matrix    
  % Add scalar value to an matrix
  new_mat = mat + 2;   % new_mat & mat are two matrix of m x n size
  
  % Add matrix A with matrix B
  new_mat = A + B;     % new_mat, A & B are same dimension m x n matrix
  
  %%
  % Multiplication Operation on Matrix  
  % Multiply scalar value with matrix
  new_mat = mat * 2;
  
  % Multiply matrix A with matrix B
  new_mat = A * B;   % dimension of matrix A: m1 x n | dimension of matrix 
                     % B: n x m2  | dimension of matrix new_mat: m1 x m2
  
  % Multiply the corresponding elements of two matrices or vectors using 
  % the .* operator
  new_mat = A .* B;   % dimension of matrix A: m1 x n | dimension of matrix 
  %B: n x m2  | dimension of matrix new_mat: m1 x m2
  
  %%
  % Subtraction Operation on Matrix
  %  
  % Subtract scalar value to an matrix
  new_mat = mat - 2;   % new_mat & mat are two matrix of m x n size
  
  % Subtract matrix A with matrix B
  new_mat = A - B;     % new_mat, A & B are same dimension m x n matrix
  
  %%
  % Minimum Value  
  % Minimum of row or column vector
  min_value = min(vector);  
  % Minimun of m x n matrix
  min_value_vector = min(mat);    % using min() on any matrix return 
  % vector containing minimum of each column
  min_value = min(min_value_vector);
  
  % Maximum Value
  % Maximum of row or column vector
  max_value = max(vector);  
  % Maximun of m x n matrix
  max_value_vector = max(mat); % using max() on any matrix return vector 
                           % containing maximum of each column
  max_value = max(max_value_vector);
 % Median Value  
 % Median of row or column vector
  median_value = median(vector);
  
  % Median of m x n matrix
  median_value_vector = median(mat);    % using median() on any matrix 
    % return vector containing median of each column
  median_value = median(median_value_vector);  
%-------------------------------------------------------------------------
% Some nifty commands
clc Clear command window
clear Clear system memory
clear x Clear x from memory
commandwindow open/select commandwindow
whos lists data structures
whos x size, bytes, class and attributes of x
ans Last result
close all closes all figures
close(H) closes figure H
winopen(pwd) Open current folder
class(obj) returns objects class
save filename saves all variables to .mat file
save filename x,y saves x,y variables to .mat file
save -append filename x appends x to .mat file
load filename loads all variables from .mat file
ver Lists version and toolboxes
beep Makes the beep sound
doc function Help/documentation for function
docsearch string search documentation
web google.com opens webadress
inputdlg Input dialog box
methods(A) list class methods for A

%-------------------------------------------------------------------------
% Statistical commands
distrnd random numbers from dist
distpdf pdf from dist
distcdf cdf dist
distrnd random numbers from dist
hist(x) histogram of x
histfit(x) histogram and
*Standard distributions (dist): norm, t, f, gam, chi2, bino
*Standard functions: mean,median,var,cov(x,y),corr(x,y),
*quantile(x,p) is not textbook version.
(It uses interpolation for missing quantiles.

%-------------------------------------------------------------------------
% Keyboard shortcuts
edit filename Opens filename in editor
Alt Displays hotkeys
F1 Help/documentation for highlighted function
F5 Run code
F9 Run highlighted code
F10 Run code line
F11 Run code line, enter functions
Shift+F5 Leave debugger
F12 Insert break point
Ctrl+Page up/down Moves between tabs
Ctrl+shift Moves between components
Ctrl+C Interrupts code
Ctrl+D Open highlighted codes file
Ctrl+ R/T Comment/uncomment line
Ctrl+N New script
Ctrl+W Close script
Ctrl+shift+d Docks window
Ctrl+shift+u Undocks window
Ctrl+shift+m max window/restore size
%-------------------------------------------------------------------------
% Built in functions/constants
abs(x) absolute value
pi 3:1415:::
inf 1
eps floating point accuracy
1e6 106
sum(x) sums elements in x
cumsum(x) Cummulative sum
prod Product of array elements
cumprod(x) cummulative product
diff Difference of elements
round/ceil/fix/floor Standard functions..
*Standard functions: sqrt, log, exp, max, min, Bessel
*Factorial(x) is only precise for x < 21
%-------------------------------------------------------------------------

%Cell commands A cell can contain any variable type.
x=cell(a,b) a ×b cell array
x{n,m} access cell n,m
cell2mat(x) transforms cell to matrix
cellfun(’fname’,C) Applies fname to cells in C
cellfun

%-------------------------------------------------------------------------
% Strings and regular expressions
strcomp compare strings (case sensitive)
strcompi compare strings (not case sensitive)
strncomp as strcomp, but only n first letters
strfind find string within a string, gives start position
regexp Search for regular expression
%-------------------------------------------------------------------------
%Logical operators
&& Short-Circuit AND.
& AND
|| Short-Circuit or
| or
~ not
== Equality comparison
~= not equal
isa(obj, ’class_name’) is object in class
*Other logical operators: <,>,>=,<=
*All above operators are elementwise
*Class indicators: isnan, isequal, ischar, isinf, isvector
, isempty, isscalar, iscolumn
*Short circuits only evaluate second criteria if
first criteria is passed, it is therefore faster.
And useful fpr avoiding errors occuring in second criteria
*non-SC are bugged and short circuit anyway
%-------------------------------------------------------------------------
%Variable generation
j:k row vector [j,j+1,...,k]
j:i:k row vector [j,j+i,...,k],
linspace(a,b,n) n points linearly spaced
and including a and b
NaN(a,b) a×b matrix of NaN values
ones(a,b) a×b matrix of 1 values
zeros(a,b) a×b matrix of 0 values
meshgrid(x,y) 2d grid of x and y vectors
[a,b]=deal(NaN(5,5)) declares a and b
global x gives x global scope
%-------------------------------------------------------------------------
% Tables
T=table(var1,var2,...,varN) Makes table*
T(rows,vars) get sub-table
T{rows,vars} get data from table
T.var or T.(varindex) all rows of var
T.var(rows) get values of var from rows
summary(T) summary of table
T.var3(T.var3>5)=5 changes some values
T.Properties.Varnames Variable names
T = array2table(A) ! make table from array
T = innerjoin(T1,T2) innerjoin
T = outerjoin(T1,T2) outerjoin !
Rows and vars indicate rows and variables.
tables are great for large datasets, because they
use less memory and allow faster operations.
*rowfun is great for tables, much faster than eg. looping
%-------------------------------------------------------------------------
% matrix and vector operations/functions
x=[1, 2, 3] 1x3 (Row) vector
x=[1; 2; 3] 3x1 (Column) vector
x=[1, 2; 3, 4] 2x2 matrix
x(2)=4 change index value nr 2
x(:) All elements of x (same as x)
x(j:end) j’th to last element of x
x(2:5) 2nd to 5th element of x
x(j,:) all j row elements
x(:,j) all j column elements
diag(x) diagonal elements of x
x.*y Element by element multiplication
x./y Element by element division
x+y Element by element addition
x-y Element by element subtraction
A^n normal/Matrix power of A
A.^n Elementwise power of A
A’ Transpose
inv(A) Inverse of matrix
size(x) Rows and Columns
eye(n) Identity matrix
sort(A) sorts vector from smallest to largest
eig(A) Eigenvalues and eigenvectors
numel(A) number of array elements
x(x>5)=0 change elemnts >5 to 0
x(x>5) list elements >5
find(A>5) Indices of elements >5
find(isnan(A)) Indices of NaN elements
[A,B] concatenates horizontally
[A;B] concatenates vertically
For functions on matrices, see bsxfun,arrayfun or repmat
*if arrayfun/bsxfun is passed a gpuArray, it runs on GPU.
*Standard operations: rank,rref,kron,chol
*Inverse of matrix inv(A) should almost never be used, use RREF
through n instead: inv(A)b = Anb.
%-------------------------------------------------------------------------
% Plotting commands
fig1 = plot(x,y) 2d line plot, handle set to fig1
set(fig1, ’LineWidth’, 2) change line width
set(fig1, ’LineStyle’, ’-’) dot markers (see *)
set(fig1, ’Marker’, ’.’) marker type (see *)
set(fig1, ’color’, ’red’) line color (see *)
set(fig1, ’MarkerSize’, 10) marker size (see *)
set(fig1, ’FontSize’, 14) fonts to size 14
figure new figure window
figure(j) graphics object j
get(j) returns information
graphics object j
gcf(j) get current figure handle
subplot(a,b,c) Used for multiple
figures in single plot
xlabel(’\mu line’,’FontSize’,14) names x/y/z axis
ylim([a b]) Sets y/x axis limits
for plot to a-b
title(’name’,’fontsize’,22) names plot
grid on/off; Adds grid to plot
legend(’x’,’y’,’Location’,’Best’) adds legends
hold on retains current figure
when adding new stuff
hold off restores to default
(no hold on)
set(h,’WindowStyle’,’Docked’); Docked window
style for plots
datetick(’x’,yy) time series axis
plotyy(x1,y1,x2,y2) plot on two y axis
refreshdata refresh data in graph
if specified source
drawnow do all in event queue
* Some markers: ’, +, *, x, o, square
* Some colors: red, blue, green, yellow, black
* color shortcuts: r, b, g, y, k
* Some line styles: -, --, :, -.
* shortcut combination example: plot(x,y,’b--o’)
%-------------------------------------------------------------------------
% Nonlinear nummerical methods
quad(fun,a,b) simpson integration of @fun
from a to b
fminsearch(fun,x0) minimum of unconstrained
multivariable function
using derivative-free method
fmincon minimum of constrained function
Example: Constrained log-likelihood maximization, note the -
Parms_est = fmincon(@(Parms) -flogL(Parms,x1,x2,x3,y)
,InitialGuess,[],[],[],[],LwrBound,UprBound,[]);
%-------------------------------------------------------------------------
% Debbuging etc.
keyboard Pauses exceution
return resumes exceution
tic starts timer
toc stops timer
profile on starts profiler
profile viewer Lets you see profiler output
try/catch Great for finding where
errors occur
dbstop if error stops at first
error inside try/catch block
dbclear clears breakpoints
dbcont resume execution
lasterr Last error message
lastwarn Last warning message
break Terminates executiion of for/while loop
waitbar Waiting bar

%-------------------------------------------------------------------------
% Data import/export
xlsread/xlswrite Spreadsheets (.xls,.xlsm)
readtable/writetable Spreadsheets (.xls,.xlsm)
dlmread/dlmwrite text files (txt,csv)
load/save -ascii text files (txt,csv)
load/save matlab files (.m)
imread/imwrite Image files

%-------------------------------------------------------------------------
% Programming commands
return Return to invoking function
exist(x) checks if x exists
G=gpuArray(x) Convert varibles to GPU array
function [y1,...,yN] = myfun(x1,...,xM)
Anonymous functions not stored in main programme
myfun = @(x1,x2) x1+x2;
or even using
myfun2 = @myfun(x) myfun(x3,2)

%-------------------------------------------------------------------------
% Conditionals and loops
for i=1:n
    procedure Iterates over procedure
end incrementing i from 1 to n by 1

while(criteria)
    procedure Iterates over procedure
end as long as criteria is true(1)
%-------------------------------------------------------------------------
if(criteria 1) if criteria 1 is true do procedure 1
    procedure1
elseif(criteria 2) ,else if criteria 2 is true do procedure 2
    procedure2
else , else do procedure 3
    procedure3
end
%-------------------------------------------------------------------------
switch switch_expression if case n holds,
    case 1 run procedure n. If none holds
        procedure 1 run procedure 3
    case 2 (if specified)
        procedure 2
   otherwise
        procedure 3
end
%-------------------------------------------------------------------------
%General comments
• Monte-Carlo: If sample sizes are increasing generate largest
size first in a vector and use increasingly larger portions for
calculations. Saves time+memory.
• Trick: Program that (1) takes a long time to run and (2)
doesnt use all of the CPU/memory ? - split it into more
programs and run using different workers (instances).
• Matlab is a column vector based language, load memory
columnwise first always. For faster code also prealocate
memory for variables, Matlab requires contiguous memory
usage!. Matlab uses copy-on-write, so passing pointers
(adresses) to a function will not speed it up. Change
variable class to potentially save memory (Ram) using:
int8, int16, int32, int64, double, char, logical, single
• You can turn the standard (mostly) Just-In-Time
compilation off using: feature accel off. You can use
compiled (c,c++,fortran) functions using MEX functions.
• Avoid global variables, they user-error prone and compilers
cant optimize them well.
• Functions defined in a .m file is only available there.
Preface function names with initials to avoid clashes, eg.
MrP function1.
• Graphic cards(GPU)’s have many (small) cores. If (1)
program is computationally intensive (not spending much
time transfering data) and (2) massively parallel, so
computations can be independent. Consider using the GPU!
• Using multiple cores (parallel computing) is often easy to
implement, just use parfor instead of for loops.
• Warnings: empty matrices are NOT overwritten ([] + 1 = []).
Rows/columns are added without warning if you write in a
nonexistent row/column. Good practise: Use 3i rather than
3*i for imaginary number calculations, because i might have
been overwritten by earlier. 1/0 returns inf, not NaN. Dont
use == for comparing doubles, they are floating point
precision for example: 0:01 == (1 − 0:99) = 0.
%-------------------------------------------------------------------------
%                                 ~END~
%-------------------------------------------------------------------------

