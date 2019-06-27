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

