In [ ]:

```
#hide
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```

In [ ]:

```
#hide
from fastbook import *
from fastai.vision.widgets import *
```

## From Model to Production

## **从模型到产品化**

The six lines of code we saw in <> are just one small part of the process of using deep learning in practice. In this chapter, we're going to use a computer vision example to look at the end-to-end process of creating a deep learning application. More specifically, we're going to build a bear classifier! In the process, we'll discuss the capabilities and constraints of deep learning, explore how to create datasets, look at possible gotchas when using deep learning in practice, and more. Many of the key points will apply equally well to other deep learning problems, such as those in <>. If you work through a problem similar in key respects to our example problems, we expect you to get excellent results with little code, quickly.

Let's start with how you should frame your problem.

在对于深度学习的使用中，上面这6行代码只是一小部分。在这个章节，我们将通过一个计算机视觉的案例来研究如何搭建一个深度学习的应用。更确切的说，我们要建造一个“熊熊识别器”！在这个过程中，我们将探讨深度学习的能力和局限，以及其他更多方面。这其中许多重要的观点都同样能很好的应用在其他深度学习的问题中，就好比<>中的那些。如果你正遇到和我们相似的状况，我们希望你能够用很少的代码快速得到很好的结果。

让我们从你该如何一步一步解决你的问题开始。

### The Practice of Deep Learning

### 深度学习的使用之道

We've seen that deep learning can solve a lot of challenging problems quickly and with little code. As a beginner, there's a sweet spot of problems that are similar enough to our example problems that you can very quickly get extremely useful results. However, deep learning isn't magic! The same 6 lines of code won't work for every problem anyone can think of today. Underestimating the constraints and overestimating the capabilities of deep learning may lead to frustratingly poor results, at least until you gain some experience and can solve the problems that arise. Conversely, overestimating the constraints and underestimating the capabilities of deep learning may mean you do not attempt a solvable problem because you talk yourself out of it.

我们已经知道深度学习能让我们用很少的代码快速解决一系列难题。包括我们的案例，这些难题都有这个相似突破点，作为新手的你可以利用它快速得到一个有用的结果。但是，机器学习不是万能的！诸如这6行代码不可能解决如今所有问题。至少在你有一些经验或者能够解决一些问题之前，都不要低估机器学习的局限或是高估它的能力，不然你可能会得到追悔莫及的糟糕结果。不过上述想法也可能只是你拒绝解决问题的借口。

 

We often talk to people who underestimate both the constraints and the capabilities of deep learning. Both of these can be problems: underestimating the capabilities means that you might not even try things that could be very beneficial, and underestimating the constraints might mean that you fail to consider and react to important issues.

我们经常会遇到一些人，他们认为深度学习即没什么局限也没什么作用。这都是不对的：低估深度学习的能力意味着你可能都没尝试过有益的使用方法，而低估它的局限性则意味着你没有很好的思考和处理重要的问题。

 

The best thing to do is to keep an open mind. If you remain open to the possibility that deep learning might solve part of your problem with less data or complexity than you expect, then it is possible to design a process where you can find the specific capabilities and constraints related to your particular problem as you work through the process. This doesn't mean making any risky bets — we will show you how you can gradually roll out models so that they don't create significant risks, and can even backtest them prior to putting them in production.

最好的做法是别抱任何成见。也许深度学习可以更简单高效地处理你的一部分问题，如果你愿意尝试，可能在过程中就能找到针对你的问题的能力和局限。这不是说让你去冒险—我们会告诉你如何一步步建立起模型，让他们不会造成严重风险。甚至在真正使用之前我们还可以再次测试他们。

### Starting Your Project

### **开始你的项目**

So where should you start your deep learning journey? The most important thing is to ensure that you have some project to work on—it is only through working on your own projects that you will get real experience building and using models. When selecting a project, the most important consideration is data availability. Regardless of whether you are doing a project just for your own learning or for practical application in your organization, you want something where you can get started quickly. We have seen many students, researchers, and industry practitioners waste months or years while they attempt to find their perfect dataset. The goal is not to find the "perfect" dataset or project, but just to get started and iterate from there.

所以你该从哪儿开始你的深度学习之旅呢？最重要的一点是你要做项目——只有真正做过项目你真的有搭建和使用模型的经验。当你选择项目的时候，一定要考虑是否能拿到你要的数据。不论是为了自学，

还是为了企业应用，你需要找一个能够快速开始的项目。我们已经见到许多学生，研究者，从业者花费数月尝试寻找他们认为最佳的数据集。我们的目的不是为了找出“最佳的”数据集或是项目，而是现在开始，不断迭代。  



If you take this approach, then you will be on your third iteration of learning and improving while the perfectionists are still in the planning stages!

如果你采用了这种方法，那么在完美主义者还在计划阶段时，你将不断学习进步。

 

We also suggest that you iterate from end to end in your project; that is, don't spend months fine-tuning your model, or polishing the perfect GUI, or labelling the perfect dataset… Instead, complete every step as well as you can in a reasonable amount of time, all the way to the end. For instance, if your final goal is an application that runs on a mobile phone, then that should be what you have after each iteration. But perhaps in the early iterations you take some shortcuts, for instance by doing all of the processing on a remote server, and using a simple responsive web application. By completing the project end to end, you will see where the trickiest bits are, and which bits make the biggest difference to the final result.

我们还建议你在项目中端到端地迭代；也就是不要花费太多时间润色你的模型或改进GUI或找到最佳数据集……而是从头到尾都在合理的时间内完成每一步。例如你要做一个手机应用，那你的目的就是在这段时间内做出这个应用来。也许开始你能走一些捷径，比如在远程服务器上用可靠的应用完成这个项目。通过端到端的完成项目，你会找到最难做的点以及那些对结果影响最大的点。

As you work through this book, we suggest that you complete lots of small experiments, by running and adjusting the notebooks we provide, at the same time that you gradually develop your own projects. That way, you will be getting experience with all of the tools and techniques that we're explaining, as we discuss them.

在阅读过程中，我们建议你在逐步完成你的项目同时，利用我们提供的习题集完成一些小实验。这样你能够在我们讨论的同时熟悉我们阐述的工具和技术。

>s: To make the most of this book, take the time to experiment between each chapter, be it on your own project or by exploring the notebooks we provide. Then try rewriting those notebooks from scratch on a new dataset. It's only by practicing (and failing) a lot that you will get an intuition of how to train a model.

> s：为了最大程度利用这本书，在每个章节结束后请花些时间在实践上，将知识点应用在自己的项目中或是钻研我们提供的习题集。然后再抓取一些新的数据集重新完成习题。只有通过不断的实践和失败你才能形成训练模型的直觉。

By using the end-to-end iteration approach you will also get a better understanding of how much data you really need. For instance, you may find you can only easily get 200 labeled data items, and you can't really know until you try whether that's enough to get the performance you need for your application to work well in practice.

在使用端到端迭代的方法时你也会对数据量的真实需求有一个更好的认识。例如你可能发现得到了200个被标记的数据元素非常轻松，直到在尝试使用时你才能知道这些能否让你得到需要的结果。

 

In an organizational context you will be able to show your colleagues that your idea can really work by showing them a real working prototype. We have repeatedly observed that this is the secret to getting good organizational buy-in for a project.

在工作中，一个真正有用的原型能够向你的同事证明你的想法是可行的。我们反复观察到，这是获得公司支持的秘籍。

 

Since it is easiest to get started on a project where you already have data available, that means it's probably easiest to get started on a project related to something you are already doing, because you already have data about things that you are doing. For instance, if you work in the music business, you may have access to many recordings. If you work as a radiologist, you probably have access to lots of medical images. If you are interested in wildlife preservation, you may have access to lots of images of wildlife.

既然最容易开始的是一个有数据基础的项目，那就意味着它可能是你正在做的行当，毕竟你已经有一些相关数据了。比如，如果你从事音乐行业，你会有很多唱片；如果你是放射科医生，你会有很多医学影像。如果你热衷于野生动物保护，你可能有很多野生动物的照片。

 

Sometimes, you have to get a bit creative. Maybe you can find some previous machine learning project, such as a Kaggle competition, that is related to your field of interest. Sometimes, you have to compromise. Maybe you can't find the exact data you need for the precise project you have in mind; but you might be able to find something from a similar domain, or measured in a different way, tackling a slightly different problem. Working on these kinds of similar projects will still give you a good understanding of the overall process, and may help you identify other shortcuts, data sources, and so forth.

有时候你得有点儿创造力，比如在Kaggle比赛里找到一点儿你感兴趣的机器学习项目。有时候你还得学会妥协，要是你找不到能够精准匹配的数据，你可以找些处理相似问题的数据，他们可以是相似领域的，也可以是用于其他标准的。研究这些类似的项目也能够让你对整个过程有一个很好的认识，也能帮你找到其他捷径，或是找到数据源，等等。

 

Especially when you are just starting out with deep learning, it's not a good idea to branch out into very different areas, to places that deep learning has not been applied to before. That's because if your model does not work at first, you will not know whether it is because you have made a mistake, or if the very problem you are trying to solve is simply not solvable with deep learning. And you won't know where to look to get help. Therefore, it is best at first to start with something where you can find an example online where somebody has had good results with something that is at least somewhat similar to what you are trying to achieve, or where you can convert your data into a format similar to what someone else has used before (such as creating an image from your data).

尤其是在你刚开始接触深度学习时，涉足你不熟悉的领域以及深度学习从未应用过的领域不是明智的选择。因为如果你的模型一开始就出错了，你无法判断是模型有错误还是这个问题根本没法用深度学习解决。而且你也不知道去哪儿寻求帮助。因此，你最好选择一个有成功先例的或是有相似先例的项目，或是你能把自己的数据转换成类似于别人使用过的格式（比如用你的数据创建一张图片）。

 

Let's have a look at the state of deep learning, just so you know what kinds of things deep learning is good at right now.

我们来看看深度学习的发展状况，这样你就能知道现在的深度学习擅长解决什么问题。

### The State of Deep Learning

### **深度学习现状**

Let's start by considering whether deep learning can be any good at the problem you are looking to work on. This section provides a summary of the state of deep learning at the start of 2020.

首先让我们思考一下深度学习是否对你所研究的问题有所帮助。这一章节将总结2020年初深度学习的发展状况。

 

However, things move very fast, and by the time you read this some of these constraints may no longer exist. We will try to keep the [book's website](https://book.fast.ai/) up-to-date; in addition, a Google search for "what can AI do now" is likely to provide current information.

然而，科技发展日新月异，当你读到这篇文章时有些局限可能早已消失了。我们尽可能保证 [book's website](https://book.fast.ai/)及时更新。此外，在谷歌搜索“现在AI能做什么”也许能给你一些最新的消息。

#### Computer vision

#### **计算机视觉**

There are many domains in which deep learning has not been used to analyze images yet, but those where it has been tried have nearly universally shown that computers can recognize what items are in an image at least as well as people can—even specially trained people, such as radiologists. This is known as *object recognition*. Deep learning is also good at recognizing where objects in an image are, and can highlight their locations and name each found object. This is known as *object detection* (there is also a variant of this that we saw in <>, where every pixel is categorized based on what kind of object it is part of—this is called *segmentation*). Deep learning algorithms are generally not good at recognizing images that are significantly different in structure or style to those used to train the model. For instance, if there were no black-and-white images in the training data, the model may do poorly on black-and-white images. Similarly, if the training data did not contain hand-drawn images, then the model will probably do poorly on hand-drawn images. There is no general way to check what types of images are missing in your training set, but we will show in this chapter some ways to try to recognize when unexpected image types arise in the data when the model is being used in production (this is known as checking for *out-of-domain* data).

至今还有很多领域没有使用深度学习分析图象，但在已使用过的领域中基本都证实了计算机能够和人类一样识别出图像中的物体，甚至是放射科医生这样的专业人员，这就是物体识别。深度学习还能够识别出物体在图像中的具体位置并且标记出他们的位置并为之署名。这就是物体探测（这也是<>内容的变体：一个像素都根据其所属物体分类，这也被称作划分）。对于和训练对象差别巨大的图像，深度学习算法基本上很难识别。比如说当训练数据里没有黑白图像，那这个模型对于黑白图像的识别能力就会很差。同样的，如果训练数据里没有手绘图象，那这个模型对于手绘图像的识别能力也会很差。其实并没有什么办法来检查你的训练集里缺少了什么类型的图像，但在这一章节我们会给出一些方法来试着识别在生产过程中是否出现未知图像类型（即范围外数据的检查）。

 

One major challenge for object detection systems is that image labelling can be slow and expensive. There is a lot of work at the moment going into tools to try to make this labelling faster and easier, and to require fewer handcrafted labels to train accurate object detection models. One approach that is particularly helpful is to synthetically generate variations of input images, such as by rotating them or changing their brightness and contrast; this is called *data augmentation* and also works well for text and other types of models. We will be discussing it in detail in this chapter.

物体探测系统面临的一大挑战在于图像标记是一项缓慢而昂贵的工程。现在很对人尝试着开发一些工具使标记的过程变得更快更简单，并且减少精确模型中对于手工标记的需要。一个较为有用的方法是改变输入图像的一些参数，比如旋转图像，或是调节他们的亮度和对比度，我们把它称为增加数据，它在文本或其他模型中也同样适用。我们会在这一章节详细讨论。

 

Another point to consider is that although your problem might not look like a computer vision problem, it might be possible with a little imagination to turn it into one. For instance, if what you are trying to classify are sounds, you might try converting the sounds into images of their acoustic waveforms and then training a model on those images.

另一个需要考虑的点是，有时候你的问题看着不像是计算机视觉的范畴，这时候你就得花一点想象力把它变成计算机视觉的问题。比如你要识别的是声音，那就得将声音转换为波形图然后再训练出一个模型。

#### Text (natural language processing)

#### **文本（自然语言加工）**

Computers are very good at classifying both short and long documents based on categories such as spam or not spam, sentiment (e.g., is the review positive or negative), author, source website, and so forth. We are not aware of any rigorous work done in this area to compare them to humans, but anecdotally it seems to us that deep learning performance is similar to human performance on these tasks. Deep learning is also very good at generating context-appropriate text, such as replies to social media posts, and imitating a particular author's style. It's good at making this content compelling to humans too—in fact, even more compelling than human-generated text. However, deep learning is currently not good at generating *correct* responses! We don't currently have a reliable way to, for instance, combine a knowledge base of medical information with a deep learning model for generating medically correct natural language responses. This is very dangerous, because it is so easy to create content that appears to a layman to be compelling, but actually is entirely incorrect.

计算机很善于基于类别的长短文档，比如是不是垃圾邮件，语言的情绪（积极或是消极），作者，源网站等。人们习惯了人类天生具有的分辨能力，所以很难意识到计算机识别的复杂和细微程度，不过有趣的是深度学习在这方面的表现和人类十分相似。深度学习还能够生成特定内容的文本，比如说对于媒体报道的评论，模仿某个作家的写作风格。但是现在深度学习还无法生成正确答案！比如说基于医学知识搭建的模型还无法完全正确地用自然语言生成医学问题的回复。这种做法也很危险，毕竟很有可能生成的回复极具说服力但实际是错误的。

 

Another concern is that context-appropriate, highly compelling responses on social media could be used at massive scale—thousands of times greater than any troll farm previously seen—to spread disinformation, create unrest, and encourage conflict. As a rule of thumb, text generation models will always be technologically a bit ahead of models recognizing automatically generated text. For instance, it is possible to use a model that can recognize artificially generated content to actually improve the generator that creates that content, until the classification model is no longer able to complete its task.

另一个让人担忧的场景是社会媒体中那些有针对性的，极具说服力的回复会被空前的大量使用来散播不实言论，制造不安，引起冲突。根据经验，文本生成模型在技术上总是会比自动识别已生成文档的模型先进一点点。比如那些能够识别文本是否为人工生成的模型或许可以用来不断改进文档生成器直到识别模型再也无法识别出文本来源。

 

Despite these issues, deep learning has many applications in NLP: it can be used to translate text from one language to another, summarize long documents into something that can be digested more quickly, find all mentions of a concept of interest, and more. Unfortunately, the translation or summary could well include completely incorrect information! However, the performance is already good enough that many people are using these systems—for instance, Google's online translation system (and every other online service we are aware of) is based on deep learning.

除了这些问题，深度学习还大量应用于NLP：它能将文本翻译成另一种语言，将很长的文件总结成更易理解的片段，找到所有令人感兴趣的部分在文中被提及的地方，等等。不幸的是，这种翻译或是总结也可能包含了完全错误的信息！不过深度学习在这方面的应用已经足够好到被许多人实际使用了——比如谷歌的线上翻译系统（以及其他所有我们所知的线上服务）就是基于深度学习的应用。

#### Combining text and images

#### 文本与图像结合

The ability of deep learning to combine text and images into a single model is, generally, far better than most people intuitively expect. For example, a deep learning model can be trained on input images with output captions written in English, and can learn to generate surprisingly appropriate captions automatically for new images! But again, we have the same warning that we discussed in the previous section: there is no guarantee that these captions will actually be correct.

深度学习将文本和图像结合进一个模型的能力通常比人们预计的好很多。比如，向一个深度学习的模型输入图像可以输出英文说明，而模型通过学习可以对新的图像生成异常精准的描述。但是再次警告：没有办法可以保证这些描述是正确的。

 

Because of this serious issue, we generally recommend that deep learning be used not as an entirely automated process, but as part of a process in which the model and a human user interact closely. This can potentially make humans orders of magnitude more productive than they would be with entirely manual methods, and actually result in more accurate processes than using a human alone. For instance, an automatic system can be used to identify potential stroke victims directly from CT scans, and send a high-priority alert to have those scans looked at quickly. There is only a three-hour window to treat strokes, so this fast feedback loop could save lives. At the same time, however, all scans could continue to be sent to radiologists in the usual way, so there would be no reduction in human input. Other deep learning models could automatically measure items seen on the scans, and insert those measurements into reports, warning the radiologists about findings that they may have missed, and telling them about other cases that might be relevant.

因为这些严重问题，我们通常不建议将深度学习用作一个完全不加监管的过程，而是作为一个与使用者亲密交互的过程。比起纯手工的方式，这样的方式可以更快完成人类大量的命令。也比一个人工作更加准确。比如一个自动系统可以通过CT扫描的方式识别有潜在中风风险的人，然后发送一个高级别警告让医生赶快看这个片子。中风的有效抢救时间只有三个小时，所以这种快速的反馈可以拯救生命。但同时，所有片子会继续按照正常方式传送到放射科医生那里，所以人工投入不会减少。另一个深度学习模型能够自动测量片子中的物体，然后在报告中记录下来，提醒放射科医生不要遗漏任何东西，并且向他们提供其他可能相关的病例。

 

#### Tabular data

#### 表格数据

For analyzing time series and tabular data, deep learning has recently been making great strides. However, deep learning is generally used as part of an ensemble of multiple types of model. If you already have a system that is using random forests or gradient boosting machines (popular tabular modeling tools that you will learn about soon), then switching to or adding deep learning may not result in any dramatic improvement. Deep learning does greatly increase the variety of columns that you can include—for example, columns containing natural language (book titles, reviews, etc.), and high-cardinality categorical columns (i.e., something that contains a large number of discrete choices, such as zip code or product ID). On the down side, deep learning models generally take longer to train than random forests or gradient boosting machines, although this is changing thanks to libraries such as [RAPIDS](https://rapids.ai/), which provides GPU acceleration for the whole modeling pipeline. We cover the pros and cons of all these methods in detail in <>.

最近，有很多人在研究利用深度学习分析时间序列和表格数据。但是通常深度学习的模型只是众多模型中的一部分。如果你已经有一个基于随机预测或是梯度增长机制（一个你很快会学到的表格建模工具）的系统，那么切换或增加深度学习的应用可能无法得到明显提升。深度学习确实可以增加类目多样性——比如包含自然语言的类目（书名，评论等），或是高维数列（包含大量无序数列如邮编，商品编号）。

深度学习的缺点是深度学习模型的训练时间确实比随机预测或梯度增长机制更长，不过像 [RAPIDS](https://rapids.ai/)这样的知识库为整个建模管道提供了GPU的加速，这也让上述问题得到了改善。我们将这些方法的优劣全都写在了<>中。



#### Recommendation systems

#### 推荐系统

Recommendation systems are really just a special type of tabular data. In particular, they generally have a high-cardinality categorical variable representing users, and another one representing products (or something similar). A company like Amazon represents every purchase that has ever been made by its customers as a giant sparse matrix, with customers as the rows and products as the columns. Once they have the data in this format, data scientists apply some form of collaborative filtering to *fill in the matrix*. For example, if customer A buys products 1 and 10, and customer B buys products 1, 2, 4, and 10, the engine will recommend that A buy 2 and 4. Because deep learning models are good at handling high-cardinality categorical variables, they are quite good at handling recommendation systems. They particularly come into their own, just like for tabular data, when combining these variables with other kinds of data, such as natural language or images. They can also do a good job of combining all of these types of information with additional metadata represented as tables, such as user information, previous transactions, and so forth.

推荐系统其实只是表格数据的一种。他们包含代表用户的高基数分类变量和代表商品的的高基数分类变量。像亚马逊这样的公司就把顾客的每一笔消费都看作是一个巨大的稀疏矩阵：矩阵的横轴是顾客，纵轴是商品。一旦得到这种格式的数据，数据研究者就会应用一些相关的过滤器来填充矩阵，比如顾客A购买了1，10号商品，顾客B购买了1，2，4，10号商品，计算机就会推荐A购买2，4号商品，这正是因为深度学习模型擅长于处理高基数分类变量问题，当然也就包括了推荐系统。深度学习模型不断完善，比如表格数据可以将变量与其他数据结合，类似于自然语言和图像。这些模型还善于将这些信息和其他表格形式的元数据结合，比如用户信息，历史交易等。

However, nearly all machine learning approaches have the downside that they only tell you what products a particular user might like, rather than what recommendations would be helpful for a user. Many kinds of recommendations for products a user might like may not be at all helpful—for instance, if the user is already familiar with the products, or if they are simply different packagings of products they have already purchased (such as a boxed set of novels, when they already have each of the items in that set). Jeremy likes reading books by Terry Pratchett, and for a while Amazon was recommending nothing but Terry Pratchett books to him (see <>), which really wasn't helpful because he already was aware of these books!

但是几乎所有机器学习的方法都有一个弊端，那就是他只能告诉你一个用户可能喜欢的商品是什么，而不能告诉你到底什么建议对用户更有用。许多针对用户喜好的推荐不一定都是有用的——比如用户已经对产品很熟悉了，或者这个产品和他们购买过的仅仅是包装不同而已（比如礼盒装的小说，用户可能已经有套装中的每一本了）。Jeremy喜欢 Terry Pratchett写的书，所以一段时间内亚马逊只向他推荐 Terry Pratchett的书（就像<>），这种推荐就毫无用处因为Jeremy早就知道这些书了！

![Terry Pratchett books recommendation](file:///C:/Users/ThinkPad/Desktop/trans/images/pratchett.png)



#### Other data types

#### 其他数据类型

Often you will find that domain-specific data types fit very nicely into existing categories. For instance, protein chains look a lot like natural language documents, in that they are long sequences of discrete tokens with complex relationships and meaning throughout the sequence. And indeed, it does turn out that using NLP deep learning methods is the current state-of-the-art approach for many types of protein analysis. As another example, sounds can be represented as spectrograms, which can be treated as images; standard deep learning approaches for images turn out to work really well on spectrograms.

有时你会找到一些特定领域的数据类型和现有的类目十分匹配。比如蛋白质链看上去很像自然语言组成的文档，因为他们都是一串串有复杂关系和意义的离散标记。而且事实证明NLP深度学习方法是目前最先进的蛋白质分析方法。另一个例子是，声音可以用图谱的形式展现，这样也就可以当作图像来处理。处理图像的标准深度学习方法确实同样适用于图谱的分析。

### The Drivetrain Approach

### 动力途径

There are many accurate models that are of no use to anyone, and many inaccurate models that are highly useful. To ensure that your modeling work is useful in practice, you need to consider how your work will be used. In 2012 Jeremy, along with Margit Zwemer and Mike Loukides, introduced a method called *the Drivetrain Approach* for thinking about this issue.

有很多复杂精细的模型其实毫无用处，反而一些相对简单的模型非常实用。所以为了能够让你搭建一个实用的模型，你得想想这个模型的实际应用场景。针对这个问题，在2012年Jeremy, Margit Zwemer 和Mike Loukides引进了一种名为“传动系方法”的方式。

 

The Drivetrain Approach, illustrated in <>, was described in detail in ["Designing Great Data Products"](https://www.oreilly.com/radar/drivetrain-approach-data-products/). The basic idea is to start with considering your objective, then think about what actions you can take to meet that objective and what data you have (or can acquire) that can help, and then build a model that you can use to determine the best actions to take to get the best results in terms of your objective.

如<>中所述的传动系方法可以参考["Designing Great Data Products"](https://www.oreilly.com/radar/drivetrain-approach-data-products/)获取更多细节。其基本思想是先考虑你的目的，再考虑什么方法能够达到你的目的以及哪些你现有的或者你能得到的数据对你有用，接着才是搭建一个能做出最佳判断的模型来达成你的目标。

 

Consider a model in an autonomous vehicle: you want to help a car drive safely from point A to point B without human intervention. Great predictive modeling is an important part of the solution, but it doesn't stand on its own; as products become more sophisticated, it disappears into the plumbing. Someone using a self-driving car is completely unaware of the hundreds (if not thousands) of models and the petabytes of data that make it work. But as data scientists build increasingly sophisticated products, they need a systematic design approach.

设想一个自动驾驶的模型：你要让一辆车在没有人为干预的情况下安全的从A点开到B点。这个解决方案里极为重要的一部分是出色的保护模型，但它并不是一个独立存在；只不过随着产品变得越来越复杂，它在研究中变得越来越不突出了。那些开着无人驾驶汽车的人们完全没有意识到这其中用到了数量庞大的模型以及数据字节。但随着数据学家搭建的复杂产品越来越多，他们非常需要一个系统化的设计方法。



We use data not just to generate more data (in the form of predictions), but to produce *actionable outcomes*. That is the goal of the Drivetrain Approach. Start by defining a clear *objective*. For instance, Google, when creating their first search engine, considered "What is the user’s main objective in typing in a search query?" This led them to their objective, which was to "show the most relevant search result." The next step is to consider what *levers* you can pull (i.e., what actions you can take) to better achieve that objective. In Google's case, that was the ranking of the search results. The third step was to consider what new *data* they would need to produce such a ranking; they realized that the implicit information regarding which pages linked to which other pages could be used for this purpose. Only after these first three steps do we begin thinking about building the predictive *models*. Our objective and available levers, what data we already have and what additional data we will need to collect, determine the models we can build. The models will take both the levers and any uncontrollable variables as their inputs; the outputs from the models can be combined to predict the final state for our objective.

我们的目的不仅仅是用数据创造更多的数据（以推测的形式），我们还要生产可执行的产出物。这才是传动系模型的目的。首先请确定一个明确的目标。举个例子：谷歌在设计第一个搜索引擎时首先考虑了“一个用户在搜索框中输入的目的是什么？”，从而明确了用户谷歌的目的，即“展示更多的相关结果”。然后针对这个目的去考虑要使用什么实现手段。在谷歌的案例中，他们所使用的方法是对搜索结果进行排序。第三步是考虑要做出这个排序需要哪些新的数据。他们发现了一个隐藏的信息即页面之间的相互关联关系对这个目的的实现非常有用。在这三步做完以后我们才能去考虑搭建预测模型。我们的目的和手段，我们已有的数据以及我们需要收集的额外数据决定了我们能搭建什么样的模型。这个模型会将手段和不受控变量都作为输入项，而他的输出项将会被结合起来去预测我们的目的最终的实现情况。

 

Let's consider another example: recommendation systems. The objective of a recommendation engine is to drive additional sales by surprising and delighting the customer with recommendations of items they would not have purchased without the recommendation. The lever is the ranking of the recommendations. New data must be collected to generate recommendations that will cause new sales. This will require conducting many randomized experiments in order to collect data about a wide range of recommendations for a wide range of customers. This is a step that few organizations take; but without it, you don't have the information you need to actually optimize recommendations based on your true objective (more sales!).

还有另外一个例子：推荐系统。推荐引擎的作用是通过给客户推荐能让他们眼前一亮又未购买过的商品，来引导他们额外消费。这一算法使用的方法是给推荐商品排名，即需要通过收集新的数据来生成推荐从而促成销售。这些大量客户推荐数据则需要通过一定数量的随机试验来获得。能做到这一步的企业少之又少；但如果不做，你将无法得到有用的信息来做出能够切实促进销售目的的推荐。



Finally, you could build two models for purchase probabilities, conditional on seeing or not seeing a recommendation. The difference between these two probabilities is a utility function for a given recommendation to a customer. It will be low in cases where the algorithm recommends a familiar book that the customer has already rejected (both components are small) or a book that they would have bought even without the recommendation (both components are large and cancel each other out).

最后，你能够搭建两种购买概率的模型，分别基于能或不能看见商品推荐。其区别在于前者使用了一种能够为客户做出特定推荐的效用函数，它的算法极少会出现推荐给客户一本类似他拒绝过的书（内容都很少）或不用推荐他也会买的书（内容都很多且互相抵消）。



As you can see, in practice often the practical implementation of your models will require a lot more than just training a model! You'll often need to run experiments to collect more data, and consider how to incorporate your models into the overall system you're developing. Speaking of data, let's now focus on how to find data for your project.

如你所见，你的模型在实际的实施过程中需要的远不止训练而已。你常需要做一些实验来收集更多的数据，还需要如何将你的模型融入整合系统。说到数据，那我们就来看看怎么找到你要的数据。



### Gathering Data

### 收集数据

For many types of projects, you may be able to find all the data you need online. The project we'll be completing in this chapter is a *bear detector*. It will discriminate between three types of bear: grizzly, black, and teddy bears. There are many images on the internet of each type of bear that we can use. We just need a way to find them and download them. We've provided a tool you can use for this purpose, so you can follow along with this chapter and create your own image recognition application for whatever kinds of objects you're interested in. In the fast.ai course, thousands of students have presented their work in the course forums, displaying everything from hummingbird varieties in Trinidad to bus types in Panama—one student even created an application that would help his fiancée recognize his 16 cousins during Christmas vacation!

在很多项目里，你都能够在网上找到所有你要的数据。在这个章节里 我们要完成的项目是一个“熊熊探测器”，它将对三种不同品种的熊加以区分：灰熊，黑熊和泰迪熊。在网上能找到许多这三种熊的图片供我们使用，我们只需要找到并下载这些图片。为此我们向你提供一个工具，这样你就可以跟着这个章节描述的方法加以实践并针对任何你感兴趣的物体创建一个图像识别应用。在fast.ai课程中，无数学生讲他们的作品放在课程论坛上，他们识别的物体也千奇百怪，从Trinidad的蜂鸟品种，到巴拿马的公交车种类—有一个学生甚至为他的未婚夫做了一个应用来在圣诞节假期时区分她的16个表兄弟！

 

At the time of writing, Bing Image Search is the best option we know of for finding and downloading images. It's free for up to 1,000 queries per month, and each query can download up to 150 images. However, something better might have come along between when we wrote this and when you're reading the book, so be sure to check out the [book's website](https://book.fast.ai/) for our current recommendation.

在编写期间，必应的图片搜索是我们认为最好的寻找和下载图片的选择：每个月有最多1000次免费搜索，每次搜索可以下载最多150张图片。当然，在我们编写或你在阅读的时候可能已经出现了更好的工具，所以记得访问 [book's website](https://book.fast.ai/)查看我们最新的推荐。

>important: Keeping in Touch With the Latest Services: Services that can be used for creating datasets come and go all the time, and their features, interfaces, and pricing change regularly too. In this section, we'll show how to use the Bing Image Search API available as part of Azure Cognitive Services at the time this book was written. We'll be providing more options and more up to date information on the [book's website](https://book.fast.ai/), so be sure to have a look there now to get the most current information on how to download images from the web to create a dataset for deep learning.

> 特别说明：时刻关注最新的服务：用来创建数据集的服务时刻在更新换代，而且他们的特征，接口，价格也会经常改变。在这一小节，我们将会使用必应的图片搜索API接口，因为作为Azure认知服务，它在本书编写期是可用的。在本书网站上我们会提供更多的选择以及更多的最新信息，所以你一定要去浏览一下最新的图片下载和数据集搭建的信息。



# Clean

## 数据清洗

To download images with Bing Image Search, sign up at Microsoft for a free account. You will be given a key, which you can copy and enter in a cell as follows (replacing 'XXX' with your key and executing it):

要在必应的图片搜索引擎上下载图片，需要先注册一个微软的免费账号。然后你会拿到一个秘钥，你可以复制并输入如下代码（将'XXX'替换成你的秘钥并执行）：

In [ ]:

```
key = 'XXX'
```

Or, if you're comfortable at the command line, you can set it in your terminal with:

或者，如果你更习惯用命令行，你可以在终端执行：

```
export AZURE_SEARCH_KEY=your_key_here
```

and then restart Jupyter Notebook, type this in a cell and execute it:

然后重启Jupyter笔记本，输入如下代码并执行

```
key = os.environ['AZURE_SEARCH_KEY']
```

Once you've set `key`, you can use `search_images_bing`. This function is provided by the small `utils` class included with the notebooks online. If you're not sure where a function is defined, you can just type it in your notebook to find out:

一旦你设置了`key`的值，你就能够使用`search_images_bing`了。这个函数来自于``utils``类，在网上的记事本中可以找到。如果你不能确定一个函数在哪里定义，你可以将其输入记事本来搜索：

```python
search_images_bing
```

<function utils.search_images_bing(key, term, min_sz=128)>

```
results = search_images_bing(key, 'grizzly bear')
ims = results.attrgot('content_url')
len(ims)
```

150

We've successfully downloaded the URLs of 150 grizzly bears (or, at least, images that Bing Image Search finds for that search term). Let's look at one:

我们成功地下载了150个灰熊图片（或者起码是通过这个关键词搜索到的图片）的链接。其中一个如下所示：



```python
#hide
ims = ['http://3.bp.blogspot.com/-S1scRCkI3vY/UHzV2kucsPI/AAAAAAAAA-k/YQ5UzHEm9Ss/s1600/Grizzly%2BBear%2BWildlife.jpg']
```

```python
dest = 'images/grizzly.jpg'
download_url(ims[0], dest)
```

```python
im = Image.open(dest)
im.to_thumb(128,128)
```

![Result](data:image/png;base64,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)



This seems to have worked nicely, so let's use fastai's `download_images` to download all the URLs for each of our search terms. We'll put each in a separate folder:

看上去一切顺利，接下来我们将使用fastia的`download_images `来下载所有搜索到的图片URL。我们将它们分别存放在独立的文件夹中：

```python
bear_types = 'grizzly','black','teddy'
path = Path('bears')
```

```python
if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} bear')
        download_images(dest, urls=results.attrgot('content_url'))
```



Our folder has image files, as we'd expect:

文件夹里有了预期的图片：

```
fns = get_image_files(path)
fns
```

(#421)[Path('bears/black/00000095.jpg'),Path('bears/black/00000133.jpg'),Path('bears/black/00000062.jpg'),Path('bears/black/00000023.jpg'),Path('bears/black/00000029.jpg'),Path('bears/black/00000094.jpg'),Path('bears/black/00000124.jpg'),Path('bears/black/00000056.jpeg'),Path('bears/black/00000046.jpg'),Path('bears/black/00000045.jpg')...]

> J: I just love this about working in Jupyter notebooks! It's so easy to gradually build what I want, and check my work every step of the way. I make a *lot* of mistakes, so this is really helpful to me...

> J: 我非常喜欢用Jupyter记事本完成工作！我能够轻松搭建我想搭建的东西，并且在过程中一步步检查。我犯了很多错误，所以它确实给了我很大的帮助。

 

Often when we download files from the internet, there are a few that are corrupt. Let's check:

当我们从互联网下载文件时，时常会出现损坏的文件。所以我们需要检查：

 

```
failed = verify_images(fns)
failed
```

(#0) []

To remove all the failed images, you can use `unlink` on each of them. Note that, like most fastai functions that return a collection, `verify_images` returns an object of type `L`, which includes the `map` method. This calls the passed function on each element of the collection:

你可以执行`unlink`命令来移除损坏文件。注意，和多数``fastai``函数返回值为一个集合一样，``verify_images ``返回值是一个``L``型对象，其中包含``map``方法。这是集合内每个元素的方向函数：`

```
failed.map(Path.unlink);
```



### Sidebar: Getting Help in Jupyter Notebooks

### 边栏：在Jupyter记事本中求助

upyter notebooks are great for experimenting and immediately seeing the results of each function, but there is also a lot of functionality to help you figure out how to use different functions, or even directly look at their source code. For instance, if you type in a cell:

Jupyter记事本善于完成实验并且快速得到每个函数的结果，除此之外还有很多功能帮助你了解如何使用不同的函数，甚至直接看到他们的源代码。比如当你输入以下命令：

 ```??verify_images```

a window will pop up with:

将弹出一个窗口显示：

```
Signature: verify_images(fns)
Source:   
def verify_images(fns):
    "Find images in `fns` that can't be opened"
    return L(fns[i] for i,o in
             enumerate(parallel(verify_image, fns)) if not o)
File:      ~/git/fastai/fastai/vision/utils.py
Type:      function
```

This tells us what argument the function accepts (`fns`), then shows us the source code and the file it comes from. Looking at that source code, we can see it applies the function `verify_image` in parallel and only keeps the image files for which the result of that function is `False`, which is consistent with the doc string: it finds the images in `fns` that can't be opened.

结果显示了函数能够使用什么参数（fns），然后还显示了源代码以及来源文件。当我们看这段源代码时我们能够看到他使用了 `verify_image`函数，并且只保留了函数运行结果为``False``的图片文件，这与doc字符串是一样的：它能够找出``fns``中无法打开的图片。



Here are some other features that are very useful in Jupyter notebooks:

还有一些在使用Jupyter记事本中非常有用的特点：

·  At any point, if you don't remember the exact spelling of a function or argument name, you can press Tab to get autocompletion suggestions.

·  无论什么时候，如果你记不住函数或者参数的完成片拼写，你可以按Tab键获得自动完成的建议。

·  When inside the parentheses of a function, pressing Shift and Tab simultaneously will display a window with the signature of the function and a short description. Pressing these keys twice will expand the documentation, and pressing them three times will open a full window with the same information at the bottom of your screen.

·  在函数的插入中，同时按Shift和Tab键会弹出一个对话框展示函数的签名以及简介。按两侧可以放大文档，按三次可以打开完成的窗口显示相同信息。

·  In a cell, typing `?func_name` and executing will open a window with the signature of the function and a short description.

·  在输入框中输入`?func_name`并执行会弹出一个对话框展示函数的签名以及简介。

·  In a cell, typing `??func_name` and executing will open a window with the signature of the function, a short description, and the source code.

·  在输入框中输入`??func_name`并执行会弹出一个对话框展示函数的签名，简介，还有源代码。

·  If you are using the fastai library, we added a `doc` function for you: executing `doc(func_name)` in a cell will open a window with the signature of the function, a short description and links to the source code on GitHub and the full documentation of the function in the [library docs](https://docs.fast.ai).

·  如果你在使用fastai知识库，我们为你增加了一个doc函数：在输入框里执行`doc(func_name) `会弹出一个对话框展示函数的签名，简介， GitHub上源代码的地址以及[library docs](https://docs.fast.ai)里关于这个函数的完整文档。

·  Unrelated to the documentation but still very useful: to get help at any point if you get an error, type `%debug` in the next cell and execute to open the [Python debugger](https://docs.python.org/3/library/pdb.html), which will let you inspect the content of every variable.

·  最后一点和文档使用无关但也十分有用：出现错误时，在下一个输入框中输入 `%debug`并运行可以打开[Python debugger](https://docs.python.org/3/library/pdb.html)，能够帮助你检查每个变量的内容

### End sidebar

### 边栏结尾

One thing to be aware of in this process: as we discussed in <>, models can only reflect the data used to train them. And the world is full of biased data, which ends up reflected in, for example, Bing Image Search (which we used to create our dataset). For instance, let's say you were interested in creating an app that could help users figure out whether they had healthy skin, so you trained a model on the results of searches for (say) "healthy skin." <> shows you the kinds of results you would get.

在这个过程中需要注意一点：正如我们在<>中所讨论的，模型只能用来反应训练它的数据。这个世界充满了具有偏向性的数据，它们最终显示在例如必应图片搜索引擎的地方，而我们则会用这些数据来搭建自己的数据集。假如你有兴趣搭建一个应用来帮助人们识别他们的皮肤状况，那么你将利用“健康皮肤”的搜索结果训练你的模型，<>显示了你将得到的结果。

![End sidebar](file:///C:/Users/ThinkPad/Desktop/trans/images/healthy_skin.gif)



With this as your training data, you would end up not with a healthy skin detector, but a *young white woman touching her face* detector! Be sure to think carefully about the types of data that you might expect to see in practice in your application, and check carefully to ensure that all these types are reflected in your model's source data. footnote:[Thanks to Deb Raji, who came up with the "healthy skin" example. See her paper ["Actionable Auditing: Investigating the Impact of Publicly Naming Biased Performance Results of Commercial AI Products"](https://dl.acm.org/doi/10.1145/3306618.3314244) for more fascinating insights into model bias.]

如果这些数据成为你的训练数据，你最终得到的不是一个健康皮肤探测器，而是一个“年轻白人女性触摸脸部”探测器！所以请确保你仔细思考了你想要在你的应用中使用什么类型的数据，并且仔细检查这些类型是否能够正确反映在模型的源数据当中。注释：【感谢Deb Raji想出“健康皮肤”的例子。更多关于模型偏见的内容见论文["Actionable Auditing: Investigating the Impact of Publicly Naming Biased Performance Results of Commercial AI Products"](https://dl.acm.org/doi/10.1145/3306618.3314244) 】

 

Now that we have downloaded some data, we need to assemble it in a format suitable for model training. In fastai, that means creating an object called `DataLoaders`.

既然我们已经下载了一些数据，我们需要将它转换成适合训练模型的格式。在fastai里就是要创建一个叫做 `DataLoaders`的对象。

## From Data to DataLoaders

## 从数据到DataLoaders

`DataLoaders` is a thin class that just stores whatever `DataLoader` objects you pass to it, and makes them available as `train` and `valid`. Although it's a very simple class, it's very important in fastai: it provides the data for your model. The key functionality in `DataLoaders` is provided with just these four lines of code (it has some other minor functionality we'll skip over for now):

`DataLoaders`是一个比较小的类，他只用来存储`DataLoader `没通过的数据，然后将他们转换成``train``和``valid``。虽然他是一个很简单的类，但在``fastai``中非常重要：它为你的模型提供数据。`DataLoaders`的关键功能由以下四行代码组成（还有一些次要功能本次跳过）：

```python
class DataLoaders(GetAttr):
    def __init__(self, *loaders): self.loaders = loaders
    def __getitem__(self, i): return self.loaders[i]
    train,valid = add_props(lambda i,self: self[i])
```

> jargon: DataLoaders: A fastai class that stores multiple `DataLoader` objects you pass to it, normally a `train` and a `valid`, although it's possible to have as many as you like. The first two are made available as properties.

> 术语解释：DataLoaders：一个fastai的类，用来储存各种`DataLoader `没通过的数据，通常是一个``train``或一个``valid``，当然亦可以很多。前两个被设置为固定值。



Later in the book you'll also learn about the` Dataset` and ` Datasets` classes, which have the same relationship.

在这本书后面的内容中你也会学到`Dataset`和`Datasets`类，他们也有同样的关系。

 

To turn our downloaded data into a `DataLoaders` object we need to tell fastai at least four things:

想要将下载的数据转换成`DataLoaders`对象，我们起码要告诉fastai以下四件事：

 

- What kinds of data we are working with
- How to get the list of items
- How to label these items
- How to create the validation set
- 我们在是用什么类型的数据
- 如何获得条目列表
- 如何标记这些条目
- 如何创建有效集

 

So far we have seen a number of factory methods for particular combinations of these things, which are convenient when you have an application and data structure that happen to fit into those predefined methods. For when you don't, fastai has an extremely flexible system called the data block API. With this API you can fully customize every stage of the creation of your `DataLoaders`. Here is what we need to create a `DataLoaders` for the dataset that we just downloaded:

时至今日我们已经见到很多工厂式的方法来结合这些东西，当你刚好有适合这个已知方法的应用和数据结构，那么使用这个方法将为你带来一些便捷。如果你没有，你也可以使用fastai中一个极其灵活的系统，它叫做data block API。利用这个API接口你可以自己定制每个搭建`DataLoaders`环节。这里向你展示我们为刚才下载的数据集创建一个`DataLoaders`所需的步骤：

```python
bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
```



Let's look at each of these arguments in turn. First we provide a tuple where we specify what types we want for the independent and dependent variables:

让我们依次看一下这些参数。首先我们提供一个数组来确定我们想要什么类型的非独立变量和独立变量:

```
blocks=(ImageBlock, CategoryBlock)
```









<from here>The *independent variable* is the thing we are using to make predictions from, and the *dependent variable* is our target. In this case, our independent variables are images, and our dependent variables are the categories (type of bear) for each image. We will see many other types of block in the rest of this book.

For this `DataLoaders` our underlying items will be file paths. We have to tell fastai how to get a list of those files. The `get_image_files` function takes a path, and returns a list of all of the images in that path (recursively, by default):

```
get_items=get_image_files
```

Often, datasets that you download will already have a validation set defined. Sometimes this is done by placing the images for the training and validation sets into different folders. Sometimes it is done by providing a CSV file in which each filename is listed along with which dataset it should be in. There are many ways that this can be done, and fastai provides a very general approach that allows you to use one of its predefined classes for this, or to write your own. In this case, however, we simply want to split our training and validation sets randomly. However, we would like to have the same training/validation split each time we run this notebook, so we fix the random seed (computers don't really know how to create random numbers at all, but simply create lists of numbers that look random; if you provide the same starting point for that list each time—called the *seed*—then you will get the exact same list each time):

```
splitter=RandomSplitter(valid_pct=0.2, seed=42)
```

The independent variable is often referred to as `x` and the dependent variable is often referred to as `y`. Here, we are telling fastai what function to call to create the labels in our dataset:

```
get_y=parent_label
```

`parent_label` is a function provided by fastai that simply gets the name of the folder a file is in. Because we put each of our bear images into folders based on the type of bear, this is going to give us the labels that we need.

Our images are all different sizes, and this is a problem for deep learning: we don't feed the model one image at a time but several of them (what we call a *mini-batch*). To group them in a big array (usually called a *tensor*) that is going to go through our model, they all need to be of the same size. So, we need to add a transform which will resize these images to the same size. *Item transforms* are pieces of code that run on each individual item, whether it be an image, category, or so forth. fastai includes many predefined transforms; we use the `Resize` transform here:

```
item_tfms=Resize(128)
```

This command has given us a `DataBlock` object. This is like a *template* for creating a `DataLoaders`. We still need to tell fastai the actual source of our data—in this case, the path where the images can be found:

In [ ]: