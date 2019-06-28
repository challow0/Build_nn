# 光学字符识别（OCR）

> Marina Samuel

### 介绍

如果你的电脑可以洗碗，洗衣服，做饭和打扫房子怎么办？我想我可以肯定地说，大多数人都很乐意伸出援助之手！但是，计算机能够以与人类相同的方式执行这些任务需要什么？

著名的计算机科学家阿兰·图灵（Alan Turing）提出将图灵测试作为一种识别机器是否具有与人类无法区分的智能的方法。测试涉及人类向两个隐藏实体提出问题，一个是人类，另一个是机器，并试图确定哪个是哪个。如果询问器无法识别机器，则认为该机器具有人类智能。

虽然图灵测试是否是对智能的有效评估存在很多争议，以及我们是否可以构建这样的智能机器，但毫无疑问已经存在具有一定程度智能的机器。目前有一些软件可以帮助机器人在办公室中导航并执行小任务，或帮助那些患有阿尔茨海默氏症的人。人工智能（A.I.）的更常见示例是Google在您搜索某些关键字时估算您所寻找的内容的方式，或者Facebook决定将哪些内容添加到您的新闻Feed中的方式。

A.I.的一个众所周知的应用是光学字符识别（OCR）。 OCR系统是一种软件，可以将手写字符的图像作为输入并将其解释为机器可读文本。虽然将手写支票存入银行机器时可能不会三思而后行，但后台仍有一些有趣的工作要做。本章将研究使用人工神经网络（ANN）识别数字的简单OCR系统的工作示例。但首先，让我们建立更多的背景。

### 什么是人工智能？

虽然图灵对情报的定义听起来很合理，但在一天结束时，构成情报的内容基本上是一场哲学辩论。然而，计算机科学家已将某些类型的系统和算法分类为AI分支。每个分支用于解决某些问题。这些分支包括以下示例，以及许多其他示例：

- 基于某些预定义的世界知识的逻辑和概率推导和推理。例如：模糊推理可以帮助恒温器在检测到温度很高且气氛潮湿时决定何时打开空调

- 启发式搜索。例如通过搜索所有可能的动作并选择最能提升您位置的动作，搜索可用于在国际象棋游戏中找到最佳的下一步动作

- 机器学习（ML）与反馈模型。例如模式识别问题，如OCR。

通常，ML涉及使用大型数据集来训练系统以识别模式。可以标记训练数据集，这意味着系统的预期输出是针对给定输入指定的，或者是未标记的，意味着未指定预期输出。使用未标记数据训练系统的算法称为无监督算法，而使用标记数据训练的算法称为监督算法。存在许多用于创建OCR系统的ML算法和技术，其中ANN是一种方法。

### 人工神经网络

#### 什么是人工神经网络？

ANN是由互相连接的节点组成的结构。结构及其功能受到生物大脑中发现的神经网络的启发。Hebbian理论解释了这些网络如何通过物理改变其结构和链接强度来学习识别模式。类似地，典型的ANN（如图15.1所示）在节点之间具有连接，其具有在网络学习时更新的权重。标记为“1”的节点称为偏差。最左边的蓝色节点是输入节点，中间列包含隐藏节点，最右边的列包含输出节点。可能有许多隐藏节点列，称为隐藏层。

![1561216039083](/home/challow/.config/Typora/typora-user-images/1561216039083.png)

图15.1中所有循环节点内的值表示节点的输出。如果我们从层L中的顶部调用第n个节点的输出作为（L）并且将层L中的第i个节点和层L 1中的第j个节点之间的连接称为w（L）ji，则节点的输出a（2）2是：

![1561216314630](/home/challow/.config/Typora/typora-user-images/1561216314630.png)

其中f（.）被称为激活函数，b是偏差。激活函数是节点具有什么类型的输出的决策者。偏差是固定输出为1的附加节点，可以添加到ANN以提高其准确性。我们将在设计前馈ANN中看到有关这两者的更多细节`neural_network_design.py`

这种类型的网络拓扑称为前馈神经网络，因为网络中没有循环。具有其输出馈入其输入的节点的ANN被称为循环神经网络。有许多算法可用于训练前馈人工神经网络;一种常用的算法称为反向传播。我们将在本章中实现的OCR系统将使用反向传播。

### 我们如何使用人工神经网络？

与大多数其他ML方法一样，使用反向传播的第一步是决定如何将我们的问题转换或减少为可由ANN解决的问题。换句话说，我们如何控制输入数据以便将其输入ANN？对于我们的OCR系统，我们可以使用给定数字的像素位置作为输入。值得注意的是，通常情况下，选择输入数据格式并非如此简单。例如，如果我们分析大图像以识别其中的形状，我们可能需要预处理图像以识别其中的轮廓。这些轮廓将是输入。

一旦我们确定了输入数据格式，下一步是什么？由于反向传播是一种监督算法，因此需要使用标记数据进行训练，如什么是人工智能？因此，当将像素位置作为训练输入时，我们还必须传递相关的数字。这意味着我们必须找到或收集绘制数字和相关值的大型数据集。

下一步是将数据集划分为训练集和验证集。训练数据用于运行反向传播算法以设置ANN的权重。验证数据用于使用训练的网络进行预测并计算其准确性。如果我们在数据上比较反向传播与另一种算法的性能，我们会将数据分成50％用于训练，25％用于比较2种算法（验证集）的性能，最后25％用于测试精度。选择的算法（测试集）。由于我们没有比较算法，我们可以将25％的一组作为训练集的一部分进行分组，并使用75％的数据来训练网络，25％用于验证训练有素。

确定人工神经网络准确性的目的是双重的。首先，它是为了避免过度拟合的问题。当网络在预测训练集方面具有比验证集更高的准确度时，就会发生过度拟合。过度拟合告诉我们，所选择的训练数据不能很好地概括，需要进一步完善。其次，测试几个不同数量的隐藏层和隐藏节点的准确性有助于设计最佳的ANN大小。最佳ANN大小将具有足够的隐藏节点和层以进行准确预测，但也尽可能少的节点/连接以减少可能减慢训练和预测的计算开销。一旦确定了最佳尺寸并且网络已经过培训，就可以做出预测了！

### 设计简单OCR系统中的决策

在最后几段中，我们讨论了前馈人工神经网络的一些基础知识以及如何使用它们。现在是时候讨论我们如何构建OCR系统了。

首先，我们必须决定我们希望我们的系统能够做什么。为了简单起见，让我们允许用户绘制一个数字，并能够使用绘制的数字训练OCR系统，或者请求系统预测绘制的数字是什么。虽然OCR系统可以在一台机器上本地运行，但使用客户端-服务器设置可以提供更大的灵活性。它可以对人工神经网络进行众包培训，并允许强大的服务器处理密集计算。

我们的OCR系统将包含5个主要组件，分为5个文件。将有：

- 一个客户端(`ocr.js`)
- 一个服务端(`server.py`)
- 一个简单的用户界面(`ocr.html`)
- 通过反向传播训练的人工神经网络(`ocr.py`)
- ANN设计的脚本(`neural_network_design.py`)

用户界面将很简单：用于绘制数字的画布和用于训练ANN或请求预测的按钮。客户端将收集绘制的数字，将其转换为数组，并将其作为训练样本或预测请求传递给服务器进行处理。服务器将通过对ANN模块进行API调用来简单地路由训练或预测请求。ANN模块将在其第一次初始化时使用现有数据集训练网络。然后，它会将ANN权重保存到文件中，并在后续启动时重新加载它们。该模块是培训和预测逻辑的核心。最后，设计脚本用于尝试不同的隐藏节点计数并确定哪种方法效果最好。总之，这些部分为我们提供了一个非常简单但功能强大的OCR系统。

现在我们已经考虑过系统如何在高层运行，是时候把概念放到代码中了！

### 一个简单的界面(``ocr.html``)

如前所述，第一步是收集用于训练网络的数据。我们可以将一系列手写数字上传到服务器，但那会很尴尬。相反，我们可以让用户使用HTML画布实际手写页面上的数字。然后，我们可以为他们提供几个选项来训练或测试网络，在那里训练网络还涉及指定绘制的数字。这样，通过将人们指向网站以接收他们的输入，可以轻松地外包数据收集。这是一些让我们入门的HTML。

```html
<html>
<head>
    <script src="ocr.js"></script>
    <link rel="stylesheet" type="text/css" href="ocr.css">
</head>
<body onload="ocrDemo.onLoadFunction()">
    <div id="main-container" style="text-align: center;">
        <h1>OCR Demo</h1>
        <canvas id="canvas" width="200" height="200"></canvas>
        <form name="input">
            <p>Digit: <input id="digit" type="text"> </p>
            <input type="button" value="Train" onclick="ocrDemo.train()">
            <input type="button" value="Test" onclick="ocrDemo.test()">
            <input type="button" value="Reset" onclick="ocrDemo.resetCanvas();"/>
        </form> 
    </div>
</body>
</html>
```

### OCR客户端(`ocr.js`)

由于HTML画布上的单个像素可能难以看到，我们可以将ANN输入的单个像素表示为10x10真实像素的正方形。因此真正的画布是200x200像素，并且从ANN的角度来看它由20x20画布表示。以下变量将帮助我们跟踪这些测量结果。

```js
var ocrDemo = {
    CANVAS_WIDTH: 200,
    TRANSLATED_WIDTH: 20,
    PIXEL_WIDTH: 10, // TRANSLATED_WIDTH = CANVAS_WIDTH / PIXEL_WIDTH
```

然后我们可以在新表示中勾勒出像素，以便更容易看到。这里我们有一个由`drawGrid()`生成的蓝色网格。

```js
drawGrid: function(ctx) {
        for (var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH; 
                 x < this.CANVAS_WIDTH; x += this.PIXEL_WIDTH, 
                 y += this.PIXEL_WIDTH) {
            ctx.strokeStyle = this.BLUE;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.CANVAS_WIDTH, y);
            ctx.stroke();
        }
    },
```

我们还需要以可以发送到服务器的形式存储在网格上绘制的数据。为简单起见，我们可以有一个名为`data`的数组，它将无色的黑色像素标记为`0`，将彩色白色像素标记为`1`.我们还需要画布上的一些鼠标监听器，以便我们知道何时调用`fillSquare（）`来为像素白色着色而用户正在绘制一个数字。这些监听器应该跟踪我们是否处于绘图状态，然后调用`fillSquare（）`来做一些简单的数学运算并决定需要填充哪些像素。

```js
 onMouseMove: function(e, ctx, canvas) {
        if (!canvas.isDrawing) {
            return;
        }
        this.fillSquare(ctx, 
            e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseDown: function(e, ctx, canvas) {
        canvas.isDrawing = true;
        this.fillSquare(ctx, 
            e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseUp: function(e) {
        canvas.isDrawing = false;
    },

    fillSquare: function(ctx, x, y) {
        var xPixel = Math.floor(x / this.PIXEL_WIDTH);
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);
        this.data[((xPixel - 1)  * this.TRANSLATED_WIDTH + yPixel) - 1] = 1;

        ctx.fillStyle = '#ffffff';
        ctx.fillRect(xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH, 
            this.PIXEL_WIDTH, this.PIXEL_WIDTH);
    },
```

现在我们越来越接近多汁的东西了！我们需要一个准备将训练数据发送到服务器的功能。这里我们有一个相对直接的`train（）`函数，它对要发送的数据进行一些错误检查，将它添加到`trainArray`并通过调用`sendData（）`将其发送出去。

```js
train: function() {
        var digitVal = document.getElementById("digit").value;
        if (!digitVal || this.data.indexOf(1) < 0) {
            alert("Please type and draw a digit value in order to train the network");
            return;
        }
        this.trainArray.push({"y0": this.data, "label": parseInt(digitVal)});
        this.trainingRequestCount++;

        // Time to send a training batch to the server.
        if (this.trainingRequestCount == this.BATCH_SIZE) {
            alert("Sending training data to server...");
            var json = {
                trainArray: this.trainArray,
                train: true
            };

            this.sendData(json);
            this.trainingRequestCount = 0;
            this.trainArray = [];
        }
    },
```

值得注意的一个有趣设计是使用`trainingRequestCount`，`trainArray`和`BATCH_SIZE`。这里发生的是`BATCH_SIZE`是一个预定义的常量，表示客户端在向OCR处理的服务器发送批量请求之前将跟踪的训练数据量。批量请求的主要原因是避免一次性请求服务器多次请求。如果存在许多客户端（例如，许多用户在`ocr.html`页面上训练系统），或者客户端中存在另一个图层，该图层采用扫描的绘制数字并将其转换为像素以训练网络，则`BATCH_SIZE`为1将导致许多不必要的请求。这种方法很好，因为它为客户端提供了更大的灵活性，但是，在实践中，批处理也应该在需要时在服务器上进行。可能发生拒绝服务（DoS）攻击，其中恶意客户端故意向服务器发送许多请求以使其崩溃以使其崩溃。

我们还需要一个`test（）`函数。与`train（）`类似，它应该对数据的有效性进行简单检查并将其发送出去。但是，对于`test（）`，由于用户应该能够请求预测并立即获得结果，因此不会发生批处理。

```js
 test: function() {
        if (this.data.indexOf(1) < 0) {
            alert("Please draw a digit in order to test the network");
            return;
        }
        var json = {
            image: this.data,
            predict: true
        };
        this.sendData(json);
    },
```

最后，我们需要一些函数来发出HTTP POST请求，接收响应，并在此过程中处理任何潜在的错误。

```js
 receiveResponse: function(xmlHttp) {
        if (xmlHttp.status != 200) {
            alert("Server returned status " + xmlHttp.status);
            return;
        }
        var responseJSON = JSON.parse(xmlHttp.responseText);
        if (xmlHttp.responseText && responseJSON.type == "test") {
            alert("The neural network predicts you wrote a \'" 
                   + responseJSON.result + '\'');
        }
    },

    onError: function(e) {
        alert("Error occurred while connecting to server: " + e.target.statusText);
    },

    sendData: function(json) {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open('POST', this.HOST + ":" + this.PORT, false);
        xmlHttp.onload = function() { this.receiveResponse(xmlHttp); }.bind(this);
        xmlHttp.onerror = function() { this.onError(xmlHttp) }.bind(this);
        var msg = JSON.stringify(json);
        xmlHttp.setRequestHeader('Content-length', msg.length);
        xmlHttp.setRequestHeader("Connection", "close");
        xmlHttp.send(msg);
    }
```

### 一台服务器(`server.py`)

尽管是一个简单地传递信息的小型服务器，我们仍然需要考虑如何接收和处理HTTP请求。首先，我们需要决定使用哪种HTTP请求。在上一节中，客户端正在使用POST，但为什么我们决定使用它？由于数据正在发送到服务器，因此PUT或POST请求最有意义。我们只需要发送一个json正文并且没有URL参数。因此从理论上讲，GET请求也可以起作用，但在语义上没有意义。然而，PUT和POST之间的选择是程序员之间长期持续的争论; KNPLabs以幽默的方式总结了这些问题。

另一个考虑因素是是否将“训练”与“预测”请求发送到不同的端点（例如，`http：// localhost / train``和http：// localhost / predict`）或相同的端点，然后分别处理数据。在这种情况下，我们可以采用后一种方法，因为在每种情况下对数据所做的事情之间的差异很小，足以适应短的if语句。实际上，如果服务器要对每种请求类型进行更详细的处理，最好将这些作为单独的端点。此决定反过来影响了何时使用了服务器错误代码。例如，当有效载荷中未指定“train”或“predict”时，将发送400“Bad Request”错误。如果使用单独的端点，这将不是问题。由OCR系统在后台完成的处理可能由于任何原因而失败，如果在服务器内未正确处理，则发送500“内部服务器错误”。同样，如果端点是分开的，那么将有更多的空间来详细发送更多适当的错误。例如，确定内部服务器错误实际上是由错误请求引起的。

最后，我们需要决定何时何地初始化OCR系统。一个好的方法是在`server.py`中但在服务器启动之前初始化它。这是因为在第一次运行时，OCR系统需要在第一次启动时对一些预先存在的数据进行网络训练，这可能需要几分钟。如果服务器在此处理完成之前启动，则任何训练或预测的请求都会抛出异常，因为在给定当前实现的情况下，OCR对象尚未初始化。另一种可能的实现可能会创建一些不准确的初始ANN用于前几个查询，而新的ANN在后台进行异步训练。这种替代方法确实允许ANN立即使用，但实现更复杂，并且只有在服务器重置时才能在服务器启动时节省时间。这种类型的实现对于需要高可用性的OCR服务更有利。

在这里，我们将大部分服务器代码放在一个处理POST请求的短函数中。

```python
 def do_POST(s):
        response_code = 200
        response = ""
        var_len = int(s.headers.get('Content-Length'))
        content = s.rfile.read(var_len);
        payload = json.loads(content);

        if payload.get('train'):
            nn.train(payload['trainArray'])
            nn.save()
        elif payload.get('predict'):
            try:
                response = {
                    "type":"test", 
                    "result":nn.predict(str(payload['image']))
                }
            except:
                response_code = 500
        else:
            response_code = 400

        s.send_response(response_code)
        s.send_header("Content-type", "application/json")
        s.send_header("Access-Control-Allow-Origin", "*")
        s.end_headers()
        if response:
            s.wfile.write(json.dumps(response))
        return
```

### 设计前馈ANN(``neural_network_design.py``)

在设计前馈ANN时，我们必须考虑几个因素。第一个是使用什么激活功能。我们之前提到过激活函数作为节点输出的决策者。激活函数的决策类型将帮助我们决定使用哪一个。在我们的例子中，我们将设计一个ANN，为每个数字（0-9）输出0到1之间的值。接近1的值意味着ANN预测这是绘制的数字，而接近0的值意味着它预测不是绘制的数字。因此，我们想要一个激活函数，其输出接近0或接近1.我们还需要一个可微分的函数，因为我们需要导数用于我们的反向传播计算。在这种情况下，常用的函数是sigmoid，因为它满足这两个约束。
StatSoft提供了一个很好的常见激活函数及其属性列表。

要考虑的第二个因素是我们是否想要包含偏差。我们之前已经提到了几次偏差，但还没有真正谈到它们是什么或者为什么我们使用它们。让我们通过回到图15.1中计算节点输出的方式来尝试理解这一点。假设我们有一个输入节点和一个输出节点，我们的输出公式将是y= f（wx），其中y是输出，f（）是激活函数，w是节点之间链接的权重，和x是节点的变量输入。偏置本质上是一个节点，其输出始终为1.这会将输出公式更改为y= f（wx+b），其中b是偏置节点和下一个节点之间连接的权重。如果我们将w和b视为常数而x视为变量，那么添加偏差会给我们的线性函数输入增加一个常数f（.）。

因此，添加偏差允许y截距的移位，并且通常为节点的输出提供更大的灵活性。包含偏差通常是一种很好的做法，特别是对于具有少量输入和输出的ANN。偏差允许ANN的输出具有更大的灵活性，从而为ANN提供更大的准确空间。如果没有偏差，我们就不太可能使用ANN进行正确的预测，或者需要更多隐藏的节点来进行更准确的预测。

要考虑的其他因素是隐藏层的数量和每层隐藏节点的数量。对于具有许多输入和输出的较大人工神经网络，这些数字是通过尝试不同的值并测试网络性能来决定的。在这种情况下，通过训练给定大小的ANN并查看验证集的正确分类百分比来测量性能。在大多数情况下，单个隐藏层足以获得良好的性能，因此我们仅在此处尝试隐藏节点的数量。

```python
# Try various number of hidden nodes and see what performs best
for i in xrange(5, 50, 5):
    nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
    performance = str(test(data_matrix, data_labels, test_indices, nn))
    print "{i} Hidden Nodes: {val}".format(i=i, val=performance)
```

在这里，我们用5到50个隐藏节点初始化一个ANN，增量为5.然后调用`test（）`函数。

```python
def test(data_matrix, data_labels, test_indices, nn):
    avg_sum = 0
    for j in xrange(100):
        correct_guess_count = 0
        for i in test_indices:
            test = data_matrix[i]
            prediction = nn.predict(test)
            if data_labels[i] == prediction:
                correct_guess_count += 1

        avg_sum += (correct_guess_count / float(len(test_indices)))
    return avg_sum / 100
```

内循环计算正确分类的数量，然后将其除以最后尝试分类的数量。这给出了ANN的比率或百分比准确度。由于每次训练ANN时，其权重可能略有不同，我们在外循环中重复此过程100次，因此我们可以对该特定ANN配置的精确度进行平均。在我们的例子中，`neural_network_design.py`的示例运行如下所示：

```python
PERFORMANCE
-----------
5 Hidden Nodes: 0.7792
10 Hidden Nodes: 0.8704
15 Hidden Nodes: 0.8808
20 Hidden Nodes: 0.8864
25 Hidden Nodes: 0.8808
30 Hidden Nodes: 0.888
35 Hidden Nodes: 0.8904
40 Hidden Nodes: 0.8896
45 Hidden Nodes: 0.8928
```

从这个输出中我们可以得出结论，15个隐藏节点将是最优的。从10到15添加5个节点会使我们的准确度提高约1％，而将准确度再提高1％则需要添加另外20个节点。增加隐藏节点数也会增加计算开销。因此，需要更长时间训练更多隐藏节点的网络并进行预测。因此，我们选择使用导致精确度显着提高的最后隐藏节点数。当然，在设计人工神经网络时，计算开销没有问题是可能的，并且最重要的是拥有最准确的人工神经网络。在这种情况下，最好选择45个隐藏节点而不是15个。

