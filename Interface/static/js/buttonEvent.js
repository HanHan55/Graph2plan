var focus_circle = false;
var focus_line = false;
var focus_rect = "";
var rect_type = false;
var Type = "";
var adjust_graph = false;
var createNewLine = false;
var isTrans = 0;
var islLoadTest = 0;
var selectRect;
var startRectvalue = [-1, -1, -1, -1];
var startPoint = [-1, -1, -1, -1, -1];
var RelRectvalue = [];
$(document).ready(function () {
    start();//执行函数
    isTrans = 0;

});

function show(isShow) {
    // document.getElementById("rmlist").style.opacity = isShow;
    // document.getElementById("gooey-API").style.opacity = isShow;
    document.getElementById("leftbox").style.opacity = isShow;
    document.getElementById("rightbox").style.opacity = isShow;
    document.getElementById("listbox").style.opacity = isShow;
    document.getElementById("graphSearch").style.opacity = isShow;
    document.getElementById("Editing").style.opacity = isShow;

    document.getElementById("BedRoomVue").style.opacity = isShow;
    document.getElementById("BathRoomVue").style.opacity = isShow;
    document.getElementById("otherVue").style.opacity = isShow;
    document.getElementById("detailVue").style.opacity = isShow;
    document.getElementById("addVue").style.opacity = isShow;

}

$(document).ready(function () {
    show(0.0)
    setTimeout("show(1.0)", 12000)
    //load the start
    demo.init();
});

function start() {

    var leftsvg = document.getElementById('LeftGraphSVG');
    leftsvg.oncontextmenu = function () {
        return false;
    }

    $('#LeftGraphSVG').on('mousedown', function (e) {

        console.log("Left!");

        let selectX = e.clientX - leftsvg.getBoundingClientRect().left;
        let selectY = e.clientY - leftsvg.getBoundingClientRect().top;

        var roomSelect = -1;

        var arr, reg = new RegExp("(^| )ifSelectRoom=([^;]*)(;|$)");
        if (arr = document.cookie.match(reg)) {
            roomSelect = arr[2];
        } else {
            roomSelect = 0;
        }

        if (roomSelect == 1) {
            clearHighLight();

            var curRoom = "NULL";
            var curIndex = -1;

            arr, reg = new RegExp("(^| )RoomType=([^;]*)(;|$)");

            if (arr = document.cookie.match(reg)) {
                curRoom = arr[2];
            }

            arr, reg = new RegExp("(^| )CurNum=([^;]*)(;|$)");

            if (arr = document.cookie.match(reg)) {
                curIndex = arr[2];
            }
            var id = "TransCircle_" + curIndex + "_" + curRoom;
            // if (isTrans == 0) {
            //     document.getElementById("graphSearch").style = "display:flex;cursor: default;color: #000;text-align: center;vertical-align: middle;line-height: 26px;position: absolute;margin-left: 160px;"
            //
            // }
            CreateCircle(selectX / 2, selectY / 2, id);
            d3.select("body").select("#LeftGraphSVG").select("#" + id).attr('scalesize', 1);
            document.cookie = "ifSelectRoom=0";
            document.cookie = "RoomNum=" + (parseInt(curIndex) + 1)
        }
    })

    console.time('time');

    var model = 1;
    $.get("/index/Init/", {'start': model.toString()}, function () {
        console.log("load model success");
        console.timeEnd('time')

    })
    animateHeight(true);
    animateHeight1(true);
    animateHeight2(true);
    animateHeight3(true);
    animateHeight4(true);
}

function addLivingRoom(BtnID) {//这个加点的
    var arr, reg = new RegExp("(^| )RoomNum=([^;]*)(;|$)");
    var id = -1;
    if (arr = document.cookie.match(reg))
        id = parseInt(arr[2]);
    console.log(BtnID);
    var roomType = BtnID.split("_")[0];
    if (roomType == "BedRoom") {
        var Bedrandom = {0: "MasterRoom", 1: "SecondRoom", 2: "GuestRoom", 3: "ChildRoom", 4: "StudyRoom"};
        var rand = Math.random() * 5;

        roomType = Bedrandom[parseInt(rand)];
    }

    selectRoomType(roomType, id);
}

function clearHighLight() {
    var points = d3.select("body").select("#LeftGraphSVG").selectAll("circle").attr("stroke-width", 2);
}

function rect_clearHighLight() {
    var rects = d3.select("body").select("#LeftLayoutSVG").selectAll("rect").attr("stroke-width", 4);

}

function selectRoomType(roomType, id) {
    document.cookie = "RoomType=" + roomType;
    document.cookie = "ifSelectRoom=1";
    document.cookie = "CurNum=" + id;
    // document.cookie = "CurNum=" + id.split("_")[1];
    var arr, reg = new RegExp("(^| )ifSelectRoom=([^;]*)(;|$)");
    if (arr = document.cookie.match(reg))
        console.log(arr[2]);
}

function init() {
    d3.select('body').select('#RightSVG').selectAll('line').remove();
    d3.select('body').select('#RightSVG').selectAll('circle').remove();

    d3.select('body').select('#RightLayoutSVG').selectAll('line').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('circle').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('rect').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('polygon').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('clipPath').remove();

    // d3.select('body').select('#LeftGraphSVG').selectAll('.TransLine').remove();
    // d3.select('body').select('#LeftGraphSVG').selectAll('.TransCircle').remove();
    document.getElementById("graphSearch").style = "cursor: default;color: #000;text-align: center;vertical-align: middle;line-height: 26px;position: absolute;margin-left: 360px;";

    d3.select('body').select('#LeftLayoutSVG').selectAll('rect').remove();
    d3.select('body').select('#LeftLayoutSVG').selectAll('polygon').remove();
    d3.select('body').select('#LeftLayoutSVG').selectAll('clipPath').remove();
    d3.select('body').select('#LeftLayoutSVG').selectAll('g').remove();

}

function RightInit() {
    d3.select('body').select('#RightSVG').selectAll('line').remove();
    d3.select('body').select('#RightSVG').selectAll('circle').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('line').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('circle').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('rect').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('polygon').remove();
    d3.select('body').select('#RightLayoutSVG').selectAll('clipPath').remove();

}

function ListBox(ret, rooms) {
    var roomList = ret;
    console.log("roomList" + roomList);
    var hsList = document.getElementById('hsList');
    while (hsList.hasChildNodes()) {
        hsList.removeChild(hsList.firstChild);
    }
    for (var i = roomList.length - 1; i >= 0; i--) {
        var hs = roomList[i];
        var itembt = document.createElement('button');
        itembt.innerHTML = ret[i].split(".")[0];
        itembt.classList.add('api-title');
        itembt.classList.add('pngls');
        itembt.id = "Btn_" + ret[i];
        var itemimg = document.createElement('img');
        // itemimg.src="../static/Data/Img/52.png";
        //             itemimg.src="../static/Data/snapshot/"+ret[i];
        itemimg.src = "../static/Data/snapshot_train/" + ret[i];
        itembt.appendChild(itemimg);
        itembt.onclick = function () {
            RightInit();
            var all = document.getElementsByClassName("api-text");
            var i;
            for (i = 0; i < all.length; i++) {
                all[i].style.border = "0px";
            }
            d3.select('body').select('#LeftBaseSVG').selectAll('rect').remove();
            var parent = this.parentNode;
            parent.style.border = "2px solid #BEECFF";
            // d3.select('body').select('#LeftLayoutSVG').selectAll("svg > *").remove();
            console.time('time');
            console.log(this.id.split("_")[1]);
            CreateRightImage(this.id.split("_")[1]);
            var Rightid = this.id.split("_")[1];
            document.getElementById("transfer").onclick = function () {
                d3.select('body').select('#LeftGraphSVG').selectAll('.TransLine').remove();
                d3.select('body').select('#LeftGraphSVG').selectAll('.TransCircle').remove();
                CreateLeftGraph(rooms, Rightid);
                // d3.select("body").select("#LeftGraphSVG").select("#" + roomid).attr('scalesize',1);
                document.getElementById("graphSearch").style = "display:none;cursor: default;color: #000;text-align: center;vertical-align: middle;line-height: 26px;position: absolute;margin-left: 160px;";
                isTrans = 1;
                document.getElementById("graphdiv").style = "display:block;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;";
                document.getElementById("layoutdiv").style = "display:block;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;";
            }
            console.timeEnd('time')
        }

        var itemdiv = document.createElement('div');
        itemdiv.classList.add('api-text');
        itemdiv.appendChild(itembt);

        var itemli = document.createElement('li');
        itemli.classList.add('col-sm-12');
        itemli.appendChild(itemdiv);
        hsList.insertBefore(itemli, hsList.firstChild);
    }
    console.time('time');
    // CreateRightImage(ret[0]);
// ocument.getElementById("transfer").onclick = function () {
//         CreateLeftGraph(rooms, ret[0]);}
    console.timeEnd('time')
}

function NumSearch() {
    document.getElementById("graphdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;";
    document.getElementById("layoutdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;";

    d3.select('body').select('#LeftGraphSVG').selectAll('.TransLine').remove();
    d3.select('body').select('#LeftGraphSVG').selectAll('.TransCircle').remove();
    document.cookie = "RoomNum=0";
    init();
    d3.select('body').select('#LeftBaseSVG').selectAll('rect').remove();
    d3.select("body").select("#LeftLayoutSVG").selectAll(".windowsline").remove();
    d3.select("body").selectAll(".UserPoint").attr("fill", "#6bdb6a").attr("stroke", 0);
    var hsname = null;
    var arr, reg = new RegExp("(^| )hsname=([^;]*)(;|$)");
    if (arr = document.cookie.match(reg))
        hsname = arr[2];

    var points = d3.select('body').select('#LeftGraphSVG').selectAll('circle');

    var rooms = [];
    rooms.push(hsname);
    var obj = Num();
    rooms.push(obj.roomactarr);
    rooms.push(obj.roomexaarr);
    rooms.push(obj.roomnumarr);

    points.each(function (d, i) {
        var room = [];
        room.push(this.id);
        room.push(this.cx.animVal.value);
        room.push(this.cy.animVal.value);
        rooms.push(room);
    });
    $.get("/index/NumSearch/", {'userInfo': JSON.stringify(rooms)}, function (ret) {
        ListBox(ret, rooms);
    });
}


function roomcolor(rmcate) {
    switch (rmcate) {
        case "LivingRoom":
            var color = d3.rgb(244, 242, 229)
            break;
        case "MasterRoom":
            var color = d3.rgb(253, 244, 171)
            break;
        case "Kitchen":
            var color = d3.rgb(234, 216, 214)
            break;
        case "Bathroom":
            var color = d3.rgb(205, 233, 252);
            break;
        case "DiningRoom":
            var color = d3.rgb(244, 242, 229);
            break;
        case "ChildRoom":
            var color = d3.rgb(253, 244, 171);
            break;
        case "StudyRoom":
            var color = d3.rgb(253, 244, 171);
            break;
        case "SecondRoom":
            var color = d3.rgb(253, 244, 171);
            break;
        case "GuestRoom":
            var color = d3.rgb(253, 244, 171);
            break;
        case "Balcony":
            var color = d3.rgb(208, 216, 135);
            break;
        case "Entrance":
            var color = d3.rgb(244, 242, 229);
            break;
        case "Storage":
            var color = d3.rgb(249, 222, 189);
            break;
        case "Wall-in":
            var color = d3.rgb(202, 207, 239);
            break;
        case "External area":
            var color = d3.rgb(255, 255, 255);
            break;
        case "Exterior wall":
            var color = d3.rgb(79, 79, 79);
            break;
        case"Front door":
            var color = d3.rgb(255, 225, 25);
            break;
        case "Interior wall":
            var color = d3.rgb(128, 128, 128);
            break;
        case"Interior door":
            var color = d3.rgb(255, 255, 255);
            break;


        default:
            break
    }
    return color;
}

function CreateCircle(cx, cy, id, r) {
    if (r == undefined) {
        r = 5;
    }

    var title = id.split("_")[2];
    var circlecolor = roomcolor(title);
    d3.select('body').select('#LeftGraphSVG').append('circle')
        .attr("cx", cx)
        .attr("cy", cy)
        .attr("fill", circlecolor)
        .attr("r", r)
        .attr("stroke", "#000000")
        .attr("stroke-width", 2)
        .attr("id", id)
        .attr("class", "TransCircle")
        .on("mousedown", circle_mousedown)
        .on("mousemove", circle_mousemove)
        .on("mouseup", circle_mouseup)
        .on("dblclick", circle_dblclick)
        .append("title")//此处加入title标签
        .text(title);
}

function CreateLine(x1, y1, x2, y2, id) {
    d3.select('body').select('#LeftGraphSVG').append('line')
        .attr("x1", x1)
        .attr("y1", y1)
        .attr("x2", x2)
        .attr("y2", y2)
        .attr("stroke", "#000000")
        .attr("stroke-width", "2px")
        .attr("id", id)
        .attr("class", "TransLine")
        .on("mousedown", line_mousedown)
        .on("mouseup", line_mouseup)
}


function LoadTestBoundary(files) {
    init();
    if (islLoadTest == 1) {
        document.getElementById("BedRoomlb").innerHTML = "BedRoom";
        document.getElementById("BathRoomlb").innerHTML = "BathRoom";
        document.getElementById("otherlb").innerHTML = "Other Room Types";
        document.getElementById("detailedlb").innerHTML = "Detailed Bedroom Types";
        document.getElementById("graphdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;";
        document.getElementById("layoutdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;";

        // initVue();
    }
    d3.select('body').select('#LeftBaseSVG').selectAll("svg > *").remove();
    d3.select('body').select('#LeftGraphSVG').selectAll("svg > *").remove();
    d3.select('body').select('#LeftLayoutSVG').selectAll("svg > *").remove();
    d3.select('body').select('#RightLayoutSVG').selectAll("svg > *").remove();
    d3.select('body').select('#RightSVG').selectAll("svg > *").remove();
    document.getElementById('hsList').innerHTML = "";
    d3.select('body').select('#LeftBaseSVG').selectAll('polygon').remove();
    d3.select('body').select('#LeftBaseSVG').selectAll('line').remove();

    var file = files[0];
    console.log(file.name);
    document.cookie = "hsname=" + file.name;
    $.get("/index/LoadTestBoundary", {'testName': file.name}, function (ret) {
        var border = 4;
        islLoadTest = 1;
        var hsex = ret['exterior'];
        d3.select("#LeftBaseSVG")
            .append("polygon")
            .attr("points", hsex)
            .attr("fill", "none")
            .attr("stroke", roomcolor("Exterior wall"))
            .attr("stroke-width", border);
        var fontdoor_color = roomcolor("Front door");

        var door = ret['door'].split(",");
        d3.select('body').select('#LeftBaseSVG').append('line')
            .attr("x1", parseInt(door[0]))
            .attr("y1", door[1])
            .attr("x2", door[2])
            .attr("y2", door[3])
            .attr("stroke", fontdoor_color)
            .attr("stroke-width", border);

    })
    d3.select('body').select('#LeftBaseSVG').attr("transform", "scale(2)");
    d3.select('body').select('#LeftGraphSVG').attr("transform", "scale(2)");

    NumSearch();
}

function CreateLeftPlan(roombx, hsex, door, windows, indoor, windowsline, rmsize) {
    d3.select('body').select('#LeftBaseSVG').selectAll('rect').remove();
    d3.select('body').select('#LeftLayoutSVG').selectAll("svg > *").remove();

    var interior_color = roomcolor("Interior wall");
    var border = 4;
    console.log("CreateLeftPlan", roombx);
    for (var i = 0; i < roombx.length; i++) {
        var rx = roombx[i][0][0];
        var ry = roombx[i][0][1];
        var rw = roombx[i][0][2] - roombx[i][0][0];
        var rh = roombx[i][0][3] - roombx[i][0][1];
        var color = roomcolor(roombx[i][1][0]);
        var tooltip = d3.select("body").append("div")
            .attr("class", "tooltip") //用于css设置类样式
            .attr("opacity", 0.0).attr("id", "tooltip" + roombx[i][1][0])
            .text(roombx[i][1][0]);
        d3.select("#LeftLayoutSVG").append("rect").attr("x", rx)//每个矩形的起始x坐标
            .attr("y", ry)
            .attr("width", rw)
            .attr("height", rh)//每个矩形的高度
            .attr("stroke-width", border)//加边框厚度
            .attr("stroke", interior_color)
            .attr("fill", color)//填充颜色
            .attr("id", roombx[i][1][0] + "_" + roombx[i][2])
            .on("mousedown", rect_mousedown)
            .on("mousemove", rect_mousemove)
            .on("mouseup", rect_mouseup)
            .on("click", rect_click)
            .on("dblclick", rect_dblclick)
            .append("title")//此处加入title标签
            .text(roombx[i][1][0]);//title标签的文字

    }
    // for (var i = 0; i < indoor.length; i++) {
    //     d3.select("#LeftLayoutSVG").append("rect").attr("x", indoor[i][0])//每个矩形的起始x坐标
    //         .attr("y", indoor[i][1])
    //         .attr("width", indoor[i][2])
    //         .attr("height", indoor[i][3])//每个矩形的高度
    //         .attr("fill", roomcolor("Interior door"));//填充颜色
    // }

    d3.select("#LeftLayoutSVG")
        .append("polygon")
        .attr("points", hsex)
        .attr("fill", "none")
        .attr("stroke", roomcolor("Exterior wall"))
        .attr("stroke-width", border);
    var door = door.split(",");
    var fontdoor_color = roomcolor("Front door");
    d3.select('body').select('#LeftLayoutSVG').append('line')
        .attr("x1", parseInt(door[0]))
        .attr("y1", door[1])
        .attr("x2", door[2])
        .attr("y2", door[3])
        .attr("stroke", fontdoor_color)
        .attr("stroke-width", border);


    var wincolor = d3.rgb(195, 195, 195);
    // for (var i = 0; i < windows.length; i++) {
    //
    //     d3.select("#LeftBaseSVG").append("rect").attr("x", windows[i][0])//每个矩形的起始x坐标
    //         .attr("y", windows[i][1])
    //         .attr("width", windows[i][2])
    //         .attr("height", windows[i][3])//每个矩形的高度
    //         .attr("fill", "#ffffff")
    //         .attr("stroke",wincolor)
    //          .attr("stroke-width", 1);
    // }
//boudary clip
    //??
    // d3.select("body").select("#LeftCanvas").attr("style", "display:none");
    d3.select("#LeftLayoutSVG").append("clipPath")
        .attr("id", "clip-th")
        .append("polygon")
        .attr("points", hsex);
    // for (var i = 0; i < windows.length; i++) {
    //
    //     d3.select("#LeftLayoutSVG").append("rect").attr("x", windows[i][0])//每个矩形的起始x坐标
    //         .attr("y", windows[i][1])
    //         .attr("width", windows[i][2])
    //         .attr("height", windows[i][3])//每个矩形的高度
    //         .attr("fill", wincolor).attr("fill","#ffffff" )
    //         .attr("stroke",wincolor)
    //          .attr("stroke-width", 1);
    // }
    // for (var i = 0; i < windowsline.length; i++) {
    //     d3.select('body').select('#LeftLayoutSVG').append('line')
    //         .attr("x1", windowsline[i][0])
    //         .attr("y1", windowsline[i][1])
    //         .attr("x2", windowsline[i][2])
    //         .attr("y2", windowsline[i][3]).attr("stroke",wincolor)
    //          .attr("stroke-width", 1) .attr("class", "windowsline");
    // }

    d3.select('body').select('#LeftLayoutSVG').attr("transform", "scale(2)");
    d3.select("#LeftLayoutSVG").attr("clip-path", "url(#clip-th)");

}

function CreateRightImage(roomID) {
    $.getJSON("/index/LoadTrainHouse/", {'roomID': roomID}, function (ret) {
        //Graph edge
        for (var i = 0; i < ret['hsedge'].length; i++) {
            var roomA = ret['hsedge'][i][0];
            var roomB = ret['hsedge'][i][1];

            d3.select('body').select('#RightSVG').append('line')
                .attr("x1", ret['rmpos'][roomA][2])
                .attr("y1", ret['rmpos'][roomA][3])
                .attr("x2", ret['rmpos'][roomB][2])
                .attr("y2", ret['rmpos'][roomB][3])
                .attr("stroke", "#000000")
                .attr("stroke-width", "2px")
                .attr("id", ret['rmpos'][roomA][1] + "-" + ret['rmpos'][roomB][1])
        }
        //Graph node size
        console.log(ret['rmsize']);
        console.log(ret['rmpos']);
        //Graph node
        for (var i = 0; i < ret['rmpos'].length; i++) {
            d3.select('body').select('#RightSVG').append('circle')
                .attr("cx", ret['rmpos'][i][2])
                .attr("cy", ret['rmpos'][i][3])
                .attr("fill", roomcolor(ret['rmpos'][i][1]))
                // .attr("r", 5)
                .attr("r", ret['rmsize'] [i][0][0])

                .attr("stroke", "#000000")
                .attr("stroke-width", 2)
                .attr("id", (i + 1) + "-" + ret['rmpos'][i][1])
        }
        d3.select('body').select('#RightSVG').attr("transform", "scale(2)");

        var border = 4;
        //Layout room
        var roombx = ret["hsbox"];
        var interiorwall_color = roomcolor("Interior wall");
        for (var i = 0; i < roombx.length; i++) {

            var rx = roombx[i][0][0];
            var ry = roombx[i][0][1];
            var rw = roombx[i][0][2] - roombx[i][0][0];
            var rh = roombx[i][0][3] - roombx[i][0][1];
            var color = roomcolor(roombx[i][1][0]);

            d3.select("#RightLayoutSVG")
                .append("rect")
                .attr("x", rx)//每个矩形的起始x坐标
                .attr("y", ry)
                .attr("width", rw)
                .attr("height", rh)//每个矩形的高度
                .attr("stroke-width", 3)//加边框厚度
                .attr("stroke", interiorwall_color)
                .attr("fill", color)//填充颜色
                .attr("id", roombx[i][1][0]);
        }

        var hsex = ret["exterior"];

        //clip over boundary
        d3.select("#RightLayoutSVG").append("clipPath")
            .attr("id", "Rightclip-th")
            .append("polygon")
            .attr("points", hsex);
        d3.select("#RightLayoutSVG").attr("clip-path", "url(#Rightclip-th)");
        //Layout Boundary
        d3.select("#RightLayoutSVG")
            .append("polygon")
            .attr("points", hsex)
            .attr("fill", "none")
            .attr("stroke", roomcolor("Exterior wall"))
            .attr("stroke-width", 6);
        //door
        var door = ret['door'].split(",");

        var fontdoor_color = roomcolor("Front door");
        d3.select('body').select('#RightLayoutSVG').append('line')
            .attr("x1", door[0])
            .attr("y1", door[1])
            .attr("x2", door[2])
            .attr("y2", door[3])
            .attr("stroke", fontdoor_color)
            .attr("stroke-width", 6);
    });
    d3.select('body').select('#RightLayoutSVG').attr("transform", "scale(2)");

}

function GetEditGraph(ret) {
    var hsname = null;
    var arr, reg = new RegExp("(^| )hsname=([^;]*)(;|$)");
    if (arr = document.cookie.match(reg))
        hsname = arr[2];

    var newCircles = d3.select("body").select("#LeftGraphSVG").selectAll("circle");
    console.log(newCircles);
    var GraphNode = [];
    newCircles.each(function (d, i) {
        // console.log(this.cx.animVal.value, this.cy.animVal.value, this.id);
        var newnode = [];
        var idlist = this.id.split("_");
        newnode.push(idlist[1]);
        newnode.push(idlist[2]);
        newnode.push(this.cx.animVal.value);
        newnode.push(this.cy.animVal.value);
        console.log(this.attributes.scalesize.value);
        newnode.push(this.attributes.scalesize.value);
        GraphNode.push(newnode);
        // GraphNode.push(newnode);
    });
    var newLine = d3.select("body").select("#LeftGraphSVG").selectAll("line");
    // console.log(newLine);
    var GraphEdge = [];
    newLine.each(function (d, i) {
        var newedge = [];
        var idlist = this.id.split("_");
        newedge.push(idlist[1]);
        newedge.push(idlist[2]);
        GraphEdge.push(newedge);
    });
    var NewGraph = [];
    NewGraph.push(GraphNode);
    NewGraph.push(GraphEdge);
    if (ret != 0) {
        NewGraph.push(ret);
    }
    return NewGraph
}

function GetEditLayout() {
    var hsname = null;
    var arr, reg = new RegExp("(^| )hsname=([^;]*)(;|$)");
    if (arr = document.cookie.match(reg))
        hsname = arr[2];

    var newRects = d3.select("body").select("#LeftLayoutSVG").selectAll("rect");
    console.log("newRects", newRects);
    var LayRect = [];
    newRects.each(function (d, i) {
        var newrect = [];
        var idlist = this.id.split("_");
        newrect.push(idlist[0]);
        newrect.push(idlist[1]);
        newrect.push(this.x.animVal.value);
        newrect.push(this.y.animVal.value);
        newrect.push(this.x.animVal.value + this.width.animVal.value);
        newrect.push(this.y.animVal.value + this.height.animVal.value);
        LayRect.push(newrect);
    });
    return LayRect
}

function GraphSearch() {
    document.getElementById("graphdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;";
    document.getElementById("layoutdiv").style = "display:none;cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;";

    var hsname = null;
    var arr, reg = new RegExp("(^| )hsname=([^;]*)(;|$)");
    if (arr = document.cookie.match(reg))
        hsname = arr[2];
    NewGraph = GetEditGraph(0);
    var rooms = [];
    rooms.push(hsname);
    var obj = Num();
    var Numrooms = [];
    Numrooms.push(obj.roomactarr);
    Numrooms.push(obj.roomexaarr);
    Numrooms.push(obj.roomnumarr);
    $.get("/index/GraphSearch/", {
        'NewGraph': JSON.stringify(NewGraph),
        'userRoomID': hsname,
        'Numrooms': JSON.stringify(Numrooms),
    }, function (ret) {
        ListBox(ret, rooms)
    });
}

function CreateLeftGraph(rooms, roomID) {
    $.getJSON("/index/TransGraph/", {'userInfo': rooms.toString(), 'roomID': roomID}, function (ret) {
        //     $.getJSON("/index/TransGraph_net/", {'userInfo': rooms.toString(), 'roomID': roomID}, function (ret) {
        document.getElementById("Generate").onclick = function () {
            var AdjustNewGraph = [];
            AdjustNewGraph = GetEditGraph(ret['rmpos']);
            // NewGraph.push(ret['rmpos']);

            $.get("/index/AdjustGraph/", {
                'NewGraph': JSON.stringify(AdjustNewGraph),
                'userRoomID': rooms.toString().split(',')[0],
                'adptRoomID': roomID
            }, function (adjust_ret) {
                // console.log("ret");
                CreateLeftPlan(adjust_ret['roomret'], adjust_ret['exterior'], adjust_ret["door"], adjust_ret["windows"], adjust_ret["indoor"], adjust_ret["windowsline"]);
                d3.select('body').select('#LeftGraphSVG').selectAll('circle').attr("r", 0);
                console.log(adjust_ret['rmpos']);

                for (var i = 0; i < adjust_ret['rmpos'].length; i++) {
                    var id = null;
                    var Circlesize = null;
                    id = "TransCircle" + "_" + adjust_ret['rmpos'][i][4] + "_" + adjust_ret['rmpos'][i][1];
                    Circlesize = d3.select("body").select("#LeftGraphSVG").select("#" + id);
                    // console.log(id);
                    // console.log(adjust_ret['rmsize'][i][0]);
                    if (parseInt((adjust_ret['rmsize'][i][0])) == 0) {
                        adjust_ret['rmsize'][i][0] = 4;
                    }
                    Circlesize.attr("r", adjust_ret['rmsize'][i][0]);
                }
            });
        };

        for (var i = 0; i < ret['hsedge'].length; i++) {
            var roomA = ret['hsedge'][i][0];
            var roomB = ret['hsedge'][i][1];
            var A_B = ret['hsedge'][i][2];
            var id = "TransLine" + "_" + roomA + "_" + roomB + "_" + A_B;

            CreateLine(ret['rmpos'][roomA][2], ret['rmpos'][roomA][3],
                ret['rmpos'][roomB][2], ret['rmpos'][roomB][3], id);
        }
        for (var i = 0; i < ret['rmpos'].length; i++) {

            var id = "TransCircle" + "_" + i + "_" + ret['rmpos'][i][1];
            CreateCircle(ret['rmpos'][i][2], ret['rmpos'][i][3], id, ret['rmsize'] [i][0][0]);
            d3.select("body").select("#LeftGraphSVG").select("#" + id).attr('scalesize', 1);

        }

        document.cookie = "RoomNum=" + ret['rmpos'].length;
        NewGraph = GetEditGraph(ret['rmpos']);
        $.get("/index/AdjustGraph/", {
            'NewGraph': JSON.stringify(NewGraph),
            'userRoomID': rooms.toString().split(',')[0],
            'adptRoomID': roomID
        }, function (adjust_ret) {
            // console.log("ret");
            CreateLeftPlan(adjust_ret['roomret'], adjust_ret['exterior'], adjust_ret["door"], adjust_ret["windows"], adjust_ret["indoor"], adjust_ret["windowsline"]);
            document.getElementById("downLoad").onclick = function () {
                var arr, reg = new RegExp("(^| )hsname=([^;]*)(;|$)");
                if (arr = document.cookie.match(reg))
                    hsname = arr[2];
                console.log(focus_rect);
                if (document.getElementById("graph").checked == true)  {
                    var link = document.createElement('a');
                    link.href = "../static/" + hsname.split(".")[0] + ".mat";
                    var event = document.createEvent('MouseEvents');
                    event.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
                    link.dispatchEvent(event);
                } else {
                    console.log("editing");
                    var NewLay = [];
                    NewLay = GetEditLayout();
                    var newGraph = [];
                    newGraph = GetEditGraph(ret['rmpos']);
                    $.get("/index/Save_Editbox/", {
                        'NewLay': JSON.stringify(NewLay),
                        'NewGraph': JSON.stringify(newGraph),
                        'userRoomID': rooms.toString().split(',')[0],
                        'adptRoomID': roomID
                    }, function (flag) {

                            var link = document.createElement('a');
                            link.href = "../static/" + hsname.split(".")[0] + ".png.mat";
                            var event = document.createEvent('MouseEvents');
                            event.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
                            link.dispatchEvent(event);




                    });
                }

            }

        });

    });
    d3.select('body').select('#LeftGraphSVG').attr("transform", "scale(2)");

}

function showGraph(oCtl) {
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftGraphSVG").attr("opacity", "1.0") : d3.select("body").select("#LeftGraphSVG").attr("opacity", "0.0");
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftGraphSVG").attr("style", "position: relative;z-index:999!important;") : d3.select("body").select("#LeftGraphSVG").attr("style", "position: relative;z-index:888 !important;");
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftLayoutSVG").attr("style", "position: relative;margin-left: -259.5px;z-index:888!important;") : d3.select("body").select("#LeftLayoutSVG").attr("style", "position: relative;margin-left: -259.5px;z-index:999 !important;");
    if ($(oCtl).is(':checked')) {
        if (document.getElementById("layout").checked == true) {
            d3.select("body").select("#LeftGraphSVG").attr("display", "flex").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "1.0");
        } else {
            d3.select("body").select("#LeftGraphSVG").attr("display", "flex").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "0.0");
        }
        document.getElementById("graphimg").style = "display:inline-flex;";
        document.getElementById("graphdiv").style = "cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;"
        document.getElementById("Editing").style = "display:none;";

    } else {
        document.getElementById("graphimg").style = "display:none;";
        document.getElementById("Editing").style = "display:flex;margin-left: 140px;     margin-top: inherit;";

        // 方法一

        if (document.getElementById("layout").checked == true) {
            d3.select("body").select("#LeftGraphSVG").attr("display", "none").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;").attr("opacity", "1.0");
        } else {
            d3.select("body").select("#LeftGraphSVG").attr("display", "none").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "0.0");
        }
        document.getElementById("graphdiv").style = "cursor: default;color: #000;width: 90px;        border: 2px solid #bfbfbf;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 300px;"

    }
}

function showRoom(oCtl) {
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftLayoutSVG").attr("opacity", "1.0") : d3.select("body").select("#LeftLayoutSVG").attr("opacity", "0.0");
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftLayoutSVG").attr("style", "position: relative;margin-left: -259.5px;z-index:888!important;") : d3.select("body").select("#LeftLayoutSVG").attr("style", "position: relative;margin-left: -259.5px;z-index:888!important;");
    // $(oCtl).is(':checked') ? d3.select("body").select("#LeftGraphSVG").attr("style", "position: relative;z-index:999!important;") : d3.select("body").select("#LeftGraphSVG").attr("style", "position: relative;z-index:999!important;");
    if ($(oCtl).is(':checked')) {
        if (document.getElementById("graph").checked == true) {
            d3.select("body").select("#LeftGraphSVG").attr("display", "flex").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "1.0");
        } else {
            d3.select("body").select("#LeftGraphSVG").attr("display", "flex").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;").attr("opacity", "1.0");
        }
        document.getElementById("layoutimg").style = "display:inline-flex;";
        document.getElementById("layoutdiv").style = "cursor: default;color: #000;width: 90px;border: 2px solid #0072ca;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;"

    } else {
        if (document.getElementById("graph").checked == true) {
            d3.select("body").select("#LeftGraphSVG").attr("display", "flex").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "0.0");
        } else {
            d3.select("body").select("#LeftGraphSVG").attr("display", "none").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:999!important;");
            d3.select("body").select("#LeftLayoutSVG").attr("style", "margin-left: 128px;margin-top: 128px;position: absolute;z-index:888!important;").attr("opacity", "0.0");
        }
        document.getElementById("layoutimg").style = "display:none;";
        document.getElementById("layoutdiv").style = "cursor: default;color: #000;width: 90px;border: 2px solid #bfbfbf;border-radius: 30px;text-align: center;vertical-align: middle;line-height: 26px;height: 30px;position: absolute;margin-left: 400px;"

    }
}

function circle_mousedown() {
    console.log("circle_mousedown");

    if (createNewLine) {

        var id = "TransLine" + "_" + startPoint[0].split("_")[1] + "_" + this.id.split("_")[1] + "_0";

        if (hasLine(id)) {
            return;
        }

        //被选中的点现在也不显示是吧？这里变得颜色都一样
        var points = d3.select("body").select("#LeftGraphSVG").selectAll("circle").attr("stroke", "#000000").attr("stroke-width", 2);
        var selectPoint = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("stroke", "#000000").attr("stroke-width", 2);
        scalesize = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("scalesize");

        CreateLine(startPoint[1], startPoint[2], this.cx.animVal.value, this.cy.animVal.value, id);

        d3.select(this).remove();
        d3.select("#" + startPoint[0]).remove();
        adjust_graph = true;
        CreateCircle(startPoint[1], startPoint[2], startPoint[0], startPoint[3]);
        var start = d3.select("body").select("#LeftGraphSVG").select("#" + startPoint[0]).attr("scalesize", startPoint[4]);
        CreateCircle(this.cx.animVal.value, this.cy.animVal.value, this.id, selectPoint.attr('r'));
        var end = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("scalesize", scalesize);
        createNewLine = false;
        return;
    }

    focus_circle = true;
    var points = d3.select("body").select("#LeftGraphSVG").selectAll("circle").attr("stroke", "#000000").attr("stroke-width", 2);
    var selectPoint = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("stroke", "rgba(0,0,0,0.56)").attr("stroke-width", 2);
    var isDelete = document.querySelector('#isDelete');
    //禁用系统右键菜单
    document.oncontextmenu = function (eve) {
        return false;
    };

    if (d3.event.button == 2) {
        // var deletealert = confirm("是否删除？");
        // if (deletealert == true) {
        //     selectPoint.remove();
        //     focus_circle = false;
        //     adjust_graph = true;
        //     var pointInd = this.id.split("_")[1];
        //
        //     var lines = d3.select("body").select("#LeftGraphSVG").selectAll(".TransLine");
        //     lines.each(function (d, i) {
        //         var startPoint = this.id.split("_")[1];
        //         var endPoint = this.id.split("_")[2];
        //
        //         if (startPoint == pointInd || endPoint == pointInd) {
        //             adjust_graph = true;
        //             d3.select(this).remove();
        //         }
        //     })
        // }
        var leftsvg = document.getElementById('LeftGraphSVG');

//自定义右键菜单唤醒和关闭
        isDelete.style.left = (d3.event.clientX - 256) + 'px';
        isDelete.style.top = (d3.event.clientY) + 'px';
        isDelete.style.display = 'block';
        var pointInd = this.id.split("_")[1];

        //事件委托写法
        isDelete.onmousedown = function (eve) {

            if (eve.target.innerText == 'Delete') {
                setTimeout(function () {
                    selectPoint.remove();
                    focus_circle = false;
                    adjust_graph = true;

                    var lines = d3.select("body").select("#LeftGraphSVG").selectAll(".TransLine");
                    lines.each(function (d, i) {
                        var startPoint = this.id.split("_")[1];
                        var endPoint = this.id.split("_")[2];

                        if (startPoint == pointInd || endPoint == pointInd) {
                            adjust_graph = true;
                            d3.select(this).remove();
                        }
                    })
                }, 10);
            }
            if (eve.target.innerText == 'Scale*0.5') {
                SelectRadius = selectPoint.attr('r');
                ScaleRadius = SelectRadius * 0.5;
                selectPoint.attr('r', ScaleRadius);
                selectPoint.attr('scalesize', 0.5);
                console.log(selectPoint.attr('scalesize'));
            }
            if (eve.target.innerText == 'Scale*0.25') {
                SelectRadius = selectPoint.attr('r');
                ScaleRadius = SelectRadius * 0.25;
                selectPoint.attr('r', ScaleRadius);
                selectPoint.attr('scalesize', 0.25);
                console.log(selectPoint.attr('scalesize'));
            }
            if (eve.target.innerText == 'Scale*5') {
                SelectRadius = selectPoint.attr('r');
                ScaleRadius = SelectRadius * 5;
                selectPoint.attr('r', ScaleRadius);
                selectPoint.attr('scalesize', 5);
                console.log(selectPoint.attr('scalesize'));
            }
            if (eve.target.innerText == 'Scale*2') {
                SelectRadius = selectPoint.attr('r');
                ScaleRadius = SelectRadius * 2;
                selectPoint.attr('r', ScaleRadius);
                selectPoint.attr('scalesize', 2);
                console.log(selectPoint.attr('scalesize'));
            }

            isDelete.style.display = 'none';
        }
        $(document).click(function (e) {
            var pop = $('#isDelete')[0];
            if (e.target != pop && !$.contains(pop, e.target)) pop.style.display = 'none'
        })
    }

}

function hasLine(id) {
    var lines = d3.select(".TransLine");

    lines.each(function (d, i) {
        if (this.id == id)
            return true;
    });

    return false;
}

function circle_mousemove() {

    console.log("Move!");

    if (focus_circle) {
        var leftsvg = document.getElementById('LeftGraphSVG');
        let newX = d3.event.x - leftsvg.getBoundingClientRect().left;
        let newY = d3.event.y - leftsvg.getBoundingClientRect().top;

        // console.log(newX + " " + newY)

        var transLines = d3.select("body").select("#LeftGraphSVG").selectAll(".TransLine");

        var pointID = (this.id).split("_")[1];

        transLines.each(function (d, i) {
            var tmp_array = (this.id).split("_");

            if (tmp_array[1] == pointID) {
                d3.select(this).attr("x1", newX / 2).attr("y1", newY / 2);
            }
            if (tmp_array[2] == pointID) {
                d3.select(this).attr("x2", newX / 2).attr("y2", newY / 2);
            }
        })

        var selectPoint = d3.select("body").select("#LeftGraphSVG").select("#" + this.id)
            .attr("cx", newX / 2).attr("cy", newY / 2);
        adjust_graph = true;
        // console.log(adjust_graph, "adjust")
    }
}

function circle_mouseup() {
    focus_circle = false;
}

function circle_dblclick() {
    createNewLine = true;
    var selectPoint = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("stroke", "#d84447").attr("stroke-width", 3);

    startPoint[0] = this.id;
    startPoint[1] = this.cx.animVal.value;
    startPoint[2] = this.cy.animVal.value;
    console.log(this.r.animVal.value);
    startPoint[3] = this.r.animVal.value;
    startPoint[4] = this.attributes.scalesize.value;
}

function line_mousedown() {
    focus_line = true;
    var lines = d3.select("body").select("#LeftGraphSVG").selectAll("line").attr("stroke", "#000000")
    var selectLine = d3.select("body").select("#LeftGraphSVG").select("#" + this.id).attr("stroke", "#d83230");

    if (d3.event.button == 2) {

        //var startPoint = this.id.split("_")[1];
        //var endPoint = this.id.split("_")[2];

        //console.log(startPoint,endPoint);

        //var isStartSingle = true;
        //var isEndSingle = true;

        selectLine.remove();

        //var curlines = d3.select("body").select("#LeftGraphSVG").selectAll("line");

        /*curlines.each(function(d,i){
            var tmp_start = this.id.split("_")[1];
            var tmp_end = this.id.split("_")[2];

            if(startPoint == tmp_start || startPoint == tmp_end) isStartSingle = false;
            if(endPoint == tmp_start || endPoint == tmp_end) isEndSingle = false;
        })

        console.log(isStartSingle,isEndSingle);

        var circles = d3.select("body").select("#LeftGraphSVG").selectAll(".TransCircle");

        circles.each(function(d,i){
            var tmp_ind = this.id.split("_")[1];

            if(isStartSingle && tmp_ind==startPoint) d3.select(this).remove();
            if(isEndSingle && tmp_ind==endPoint) d3.select(this).remove();
        })*/

        focus_line = false;
    }
}

function line_mouseup() {
    focus_line = false;
}

function rect_mousedown() {
    console.log("rect_mousedown");

    if (focus_rect != "") {
        var leftlaysvg = document.getElementById('LeftLayoutSVG');

        let mousex = (d3.event.x - leftlaysvg.getBoundingClientRect().left) / 2;
        let mousey = (d3.event.y - leftlaysvg.getBoundingClientRect().top) / 2;
        var oldx = startRectvalue[0];
        var oldy = startRectvalue[1];
        var oldw = startRectvalue[2];
        var oldh = startRectvalue[3];

        Type = rectzoomType(mousex, mousey, oldx, oldy, oldw, oldh);
        console.log(Type);

        rect_type = true;
    }
}

function rectzoomType(mousex, mousey, oldx, oldy, oldw, oldh) {
    if (oldy < mousey && mousey < oldy + oldh) {
        if (mousex < oldx + oldw + 16) {
            // $('#LeftLayoutSVG').css('cursor', 'e-resize');
            if (oldx + oldw - 16 < mousex) {
                d3.select("body").select("#LeftLayoutSVG").attr('cursor', 'e-resize');
                var type = "right";
                return type;
            }
        }
        if (mousex < oldx + 12) {
            if (oldx - 16 < mousex) {
                d3.select("body").select("#LeftLayoutSVG").attr('cursor', 'w-resize');
                var type = "left";
                return type;

            }

        }
    }
    if (oldx < mousex && oldx < oldx + oldw) {
        if (mousey < oldy + 16) {
            if (oldy - 16 < mousey) {
                d3.select("body").select("#LeftLayoutSVG").attr('cursor', 'n-resize');
                var type = "top";
                return type;
            }


        }
        if (mousey < oldy + oldh + 16) {
            if (oldy + oldh - 16 < mousey) {
                d3.select("body").select("#LeftLayoutSVG").attr('cursor', 's-resize');
                var type = "down";
                return type;

            }

        }
    }
    if (type == undefined) {
        d3.select("body").select("#LeftLayoutSVG").attr('cursor', 'default');

    }
}

function rect_mousemove() {

    var leftlaysvg = document.getElementById('LeftLayoutSVG');
    let mousex = (d3.event.x - leftlaysvg.getBoundingClientRect().left) / 2;
    let mousey = (d3.event.y - leftlaysvg.getBoundingClientRect().top) / 2;
        console.log("rect_mousemove",mousex,mousey);
    console.log(focus_rect);
    if (focus_rect == "dblclick") {
        // var oldx = this.x.animVal.value;
        // var oldy = this.y.animVal.value;
        //
        // var oldw = this.width.animVal.value;
        // var oldh = this.height.animVal.value;
        var oldx = startRectvalue[0];
        var oldy = startRectvalue[1];
        var oldw = startRectvalue[2];
        var oldh = startRectvalue[3];
        rectzoomType(mousex, mousey, oldx, oldy, oldw, oldh);
        // console.log(type);
        if (rect_type) {
            var item = null;
            var obj = document.getElementsByName("edit");
            for (var i = 0; i < obj.length; i++) { //遍历Radio
                if (obj[i].checked) {
                    item = obj[i].value;
                }
            }

            if (item == "local") {
                switch (Type) {
                    case "right":
                        selectRect.attr("width", mousex - oldx);
                        break;
                    case "left":
                        // selectRect.attr("x", mousex).attr("width", mousex - oldx + oldw);
                        selectRect.attr("x", mousex);
                        selectRect.attr("width", oldx - mousex + oldw);
                        break;
                    case "top":
                        selectRect.attr("y", mousey).attr("height", oldy - mousey + oldh);
                        break;
                    case "down":
                        selectRect.attr("height", mousey - oldy);
                        break;

                }

            }
            if (item == "global") {
                switch (Type) {
                    case "right":
                        for (i = 0; i < RelRectvalue[0].length; i++) {
                            var RelRect = d3.select("body").select("#LeftLayoutSVG").select("#" + RelRectvalue[0][i][4]);
                            RelRect.attr("width", mousex - RelRectvalue[0][i][0]);
                        }
                        break;
                    case "left":
                        for (i = 0; i < RelRectvalue[1].length; i++) {
                            var RelRect = d3.select("body").select("#LeftLayoutSVG").select("#" + RelRectvalue[1][i][4]);
                            RelRect.attr("x", mousex);
                            RelRect.attr("width", Number(RelRectvalue[1][i][0]) - mousex + Number(RelRectvalue[1][i][2]));
                        }
                        break;

                    case "down":
                        for (i = 0; i < RelRectvalue[3].length; i++) {
                            var RelRect = d3.select("body").select("#LeftLayoutSVG").select("#" + RelRectvalue[3][i][4]);
                            RelRect.attr("height", mousey - RelRectvalue[3][i][1]);
                        }
                        break;

                    case "top":

                        for (i = 0; i < RelRectvalue[2].length; i++) {
                            console.log(RelRectvalue[2][i]);
                            var RelRect = d3.select("body").select("#LeftLayoutSVG").select("#" + RelRectvalue[2][i][4]);
                            RelRect.attr("y", mousey).attr("height", Number(RelRectvalue[2][i][1]) - mousey + Number(RelRectvalue[2][i][3]));
                        }
                        break;
                }

            }

        }
    }
}


function rect_mouseup() {
    console.log("rect_mouseup");
    focus_rect = "";
    rect_type = false;
    var interior_color = roomcolor("Interior wall");

    var rects = d3.select("body").select("#LeftLayoutSVG").selectAll("rect").attr("stroke", interior_color).attr("stroke-width", 4);
    d3.select("body").select("#LeftLayoutSVG").attr('cursor', 'default');

}

function rect_dblclick() {
    console.log("rect_dblclick");
    var item = null;
    var obj = document.getElementsByName("edit");
    for (var i = 0; i < obj.length; i++) { //遍历Radio
        if (obj[i].checked) {
            item = obj[i].value;
        }
    }
    var interior_color = roomcolor("Interior wall");
    var rects = d3.select("body").select("#LeftLayoutSVG").selectAll("rect").attr("stroke", interior_color).attr("stroke-width", 4);
    selectRect = d3.select("body").select("#LeftLayoutSVG").select("#" + this.id).attr("stroke", "#d84447").attr("stroke-width", 4);
    focus_rect = "dblclick";


    startRectvalue[0] = this.x.animVal.value;
    startRectvalue[1] = this.y.animVal.value;
    startRectvalue[2] = this.width.animVal.value;
    startRectvalue[3] = this.height.animVal.value;
    if (item == "global") {
        console.log(this.id);
        $.get("/index/RelBox/", {
            'selectRect': this.id

        }, function (rdirgroup) {

            for (k = 0; k < rdirgroup.length; k++) {
                var RelRectvalue2 = [];
                for (i = 0; i < rdirgroup[k].length; i++) {
                    var RelRectvalue1 = [];

                    var RelRect = d3.select("body").select("#LeftLayoutSVG").select("#" + rdirgroup[k][i]);
                    RelRectvalue1[0] = RelRect.attr("x");
                    RelRectvalue1[1] = RelRect.attr("y");
                    RelRectvalue1[2] = RelRect.attr("width");
                    RelRectvalue1[3] = RelRect.attr("height");
                    RelRectvalue1[4] = rdirgroup[k][i];
                    RelRectvalue2[i] = RelRectvalue1
                }
                RelRectvalue[k] = RelRectvalue2;
            }
            console.log(RelRectvalue);
            console.log(RelRectvalue[2]);

            console.log(RelRectvalue[0]);

        });
    }
}

function rect_click() {
    console.log("rect_click");
    focus_rect = "click";

}