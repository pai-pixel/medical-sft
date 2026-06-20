Vue.use(httpVueLoader);

new Vue({
  el: '#home',
  data: function () {
    return {
      bannerData: [
        {
          bgImg: api.imgUrl + 'yaopin_bg.jpg',
          // bgImg:'./image/yaopin_bg.jpg',
          // text: '药品',
          en: 'Drugs',
          icon: 'el-icon-yaopin hidden-xs-only',
          className: 'yaopin active',
          paraCode: 'item_2'
        },
        {
          bgImg: api.imgUrl + 'yiliaoqixie_bg.jpg',
          // bgImg: './image/yiliaoqixie_bg.jpg',
          // text: '医疗器械',
          en: 'Medical Devices',
          icon: 'el-icon-yiliaoqixie hidden-xs-only',
          className: 'yiliaoqixie',
          paraCode: 'item_3'
        },
        {
          bgImg: api.imgUrl + 'huazhuangpin_bg.jpg',
          // bgImg: './image/huazhuangpin_bg.jpg',
          // text: '化妆品',
          en: 'Cosmetics',
          icon: 'el-icon-huazhuangpin hidden-xs-only',
          className: 'huazhuangpin',
          paraCode: 'item_4'
        },
        {
          bgImg: api.imgUrl + 'qita_bg.jpg',
          // bgImg: './image/qita_bg.jpg',
          // text: '其他',
          en: 'Other',
          icon: 'el-icon-qita hidden-xs-only',
          className: 'qita',
          paraCode: 'item_6'
        }
      ],
      bannerOldIndex: 0,
      bannerIndex: 0,
      hoverTimer: '',
      // 搜索关键字
      bannerSearch: '',
      restaurants: [],
      hisArray: [],
      searchSelect: '',
      gjFeildTipTxt: '',
      // 选项数据集合
      datastructArray: [],
      dbItem: [],
      searchResultArray: [],
      searchDbResultArray: [],
      couponSelected: '',
      searchResultUpDownState: false,
      drawerShow: false,
      hotKeyArray: [],
      paraCode: '',
      paraName: '',
      itemIdArray: [],
      itemName: ''
    }
  },
  mounted: function () {
    var isintro = $.cookie("STEP_TIPS_INDEX");
    if (isintro != "true") {
      setTimeout(function () {
        $(".banner-list li a").css("color", "#4267bc");
        introJs().setOptions({
          overlayOpacity: 0.1,
          nextLabel: '下一步 &rarr;',
          prevLabel: '&larr; 上一步',
          doneLabel: "完成",
          skipLabel: '关闭'
        }).start().onbeforechange(function (targetElement) {
          if ($(targetElement).prop("id") != 'banner') {
            $(".banner-list li a").css("color", "");
          } else {
            $(".banner-list li a").css("color", "#4267bc");
          }
          //20250630解决第五个步骤的箭头指向问题
          const tooltip = document.querySelector('.introjs-tooltip');
          if (tooltip && this._currentStep === 4) {
            tooltip.setAttribute('data-step', '5'); // 标记第五步
          }
        }).onexit(function () {
          $(".banner-list li a").css("color", "");
        });
        $.cookie("STEP_TIPS_INDEX", "true", {expires: 99999, path: '/'});
      }, 700);
    }
  },
  created: function () {
    var ie = document.getElementById("ie");
    if (!!window.ActiveXObject || "ActiveXObject" in window) {
      if (ie != null) {
        var web_itemId = getUrl('itemId');
        var categoryName = getHashName('category');
        var url = "./ie-home-index.html";
        if (web_itemId) {
          url = url + "?itemId=" + web_itemId;
        }
        if (categoryName) {
          url = url + "#category=" + categoryName;
        }
        location.href = url;
      }


    }
    // 数据分类列表
    this.datastruct();

    // 根据链接锚点选择默认展示品类
    var categoryName = getHashName('category');
    // 所有className重置
    for (var i = 0; i < this.bannerData.length; i++) {
      var item = this.bannerData[i];
      item.className = item.className.replace(' active', '');
    }
    this.itemName = categoryName;
    if (categoryName) {
      switch (categoryName) {
        // 给对应className赋值
        case 'ylqx':
          this.bannerOldIndex = 1;
          this.bannerIndex = 1;
          this.bannerData[1].className += ' active';
          break;
        case 'hzp':
          this.bannerOldIndex = 2;
          this.bannerIndex = 2;
          this.bannerData[2].className += ' active';
          break;
        case 'qt':
          this.bannerOldIndex = 3;
          this.bannerIndex = 3;
          this.bannerData[3].className += ' active';
          break;

        default:
          this.bannerOldIndex = 0;
          this.bannerIndex = 0;
          this.bannerData[0].className += ' active';
      }
      var web_itemId = getUrl('itemId');
      var _this = this;
      if (web_itemId) {
        setTimeout(function () {
          _this.initWebselectKeyWord(web_itemId);
        }, 100);

      }
    } else {
      // 没有指定拼配的时候
      this.bannerOldIndex = 0;
      this.bannerIndex = 0;
      // 给对应className赋值
      this.bannerData[0].className += ' active';
    }
  },
  methods: {
    // 数据分类列表
    datastruct: function () {
      var _this = this;
      pajax.get(api.NMPA_DATA, {}).then(function (result) {
        if (result.status == 200 && result.data.length) {
          // 处理banner控制文字
          result.data.forEach(function (item, index) {
            _this.bannerData[index].text = item.paraName;
            _this.bannerData[index].id = item.id;
            //只遍历数据表
            var tables = [];
            item.itemList.forEach(function (item1, index) {
              if (item1.itemType == 'table') {
                tables[index] = item1;
              }
            })
            _this.dbItem[index] = tables;
          })
          // 取所有数据
          _this.datastructArray = result.data;
          // 默认取第一组数据
          _this.searchResultArray = result.data[_this.bannerIndex].itemList;
          _this.searchDbResultArray = _this.dbItem[_this.bannerIndex];
          _this.couponSelected = _this.dbItem[_this.bannerIndex][0].itemId;
          _this.gaojiGetJosn(_this.dbItem[_this.bannerIndex][0].itemId);
          _this.paraCode = result.data[0].paraCode;
          _this.paraName = _this.itemName == 'ylqx' ? '医疗器械' : _this.itemName == 'hzp' ? '化妆品' : _this.itemName == 'qt' ? '其它' : '药品';
          _this.searchResultArray = JSON.parse(JSON.stringify(_this.searchResultArray))
        } else {

        }
      }).catch(function (error) {

      })
    },
    // 搜索热词
    getHotKey: function () {
      var _this = this;
      pajax.hasTokenGet(api.getHotKey, {}).then(function (result) {
        var obj = result.data;

        if (obj.code == 200 && obj.data.length) {
          _this.hotKeyArray = obj.data;
        }
      }).catch(function (error) {

      })
    },

    // 搜索触发接口
    countNums: function (loading, flag) {
      var _this = this;
      pajax.hasTokenGet(api.countNums, {
        'itemIds': _this.itemIdArray.join(','),
        'searchValue': _this.bannerSearch
      }).then(function (result) {
        var obj = result.data;

        if (obj.code == '200') {
          // 储存数据
          localStorage.setItem('searchRusultArray', JSON.stringify(obj));
          localStorage.setItem('itemIdArray', JSON.stringify(_this.itemIdArray));

          // setCookie('searchRusultArray', JSON.stringify(obj));
          // setCookie('itemIdArray', JSON.stringify(_this.itemIdArray));

          // 跳转
          // window.location.href = './home.html#/search-result';
          if (flag != null && (flag == "appsearch" || flag == 'applinksearch')) {
            location.href = './app-search-result.html';
          } else {
            api.openWebWin('./search-result.html');
          }

          // searchkey='+_this.bannerSearch+'&itemId='+_this.couponSelected+'&paraCode='+_this.paraCode

          // setCookie('searchkey', _this.bannerSearch);
          localStorage.setItem('searchkey', _this.bannerSearch);

          var selectValueArray = [
            [_this.paraCode, _this.couponSelected]
          ];

          // setCookie('selectValue', JSON.stringify(selectValueArray));
          localStorage.setItem('selectValue', JSON.stringify(selectValueArray))

          // setCookie('itemId', _this.couponSelected);
          // setCookie('paraCode', _this.paraCode);
          localStorage.setItem('itemId', _this.couponSelected);
          localStorage.setItem('paraCode', _this.paraCode);
        } else {
          _this.$message({
            showClose: true,
            duration: 0,
            offset: 385,
            message: obj.message,
            type: 'error'
          });

        }
        // 取消loading
        loading.close();
      }).catch(function (reason) {
    	  console.log('服务请求异常，请重新刷新页面'+reason);
        _this.$message({
          showClose: true,
          duration: 0,
          offset: 385,
          message: '服务请求异常，请重新刷新页面',
          type: 'error'
        });
      })

    },

    hoverEvt: function (index, item) {
      //js延时防止鼠标误操作
      var _this = this;
      //this.hoverTimer = setTimeout(_this.fnBannerNav(index, item),4000);
      this.hoverTimer = setTimeout(function () {
        _this.fnBannerNav(index, item)
      }, 350);
    },
    hoverOutEvt: function (index, item) {
      //js延时防止鼠标误操作
      clearTimeout(this.hoverTimer);

    },

    // banner点击
    fnBannerNav: function (index, item) {
      var _this = this;

      // 清空下拉框选择
      this.searchSelect = '';
      this.bannerSearch = "";
      // 获取paraCode
      this.paraCode = item.paraCode;
      this.paraName = item.text;

      // 如果触发是当前元素，不做任何触发
      if (index == this.bannerOldIndex) {
        return false;
      }

      // 取触发元素索引，并存值给old
      this.bannerIndex = index;
      this.bannerOldIndex = index;

      // 清空所有元素上下划线装饰
      this.bannerData.forEach(function (item, index) {
        item.className = item.className.replace('active', '');
      })

      switch (index) {
        case 0:
          window.location.hash = 'category=yp';
          break;
        case 1:
          window.location.hash = 'category=ylqx';
          break;
        case 2:
          window.location.hash = 'category=hzp';
          break;
        case 3:
          window.location.hash = 'category=qt';
          break;
      }

      // 为触发元素添加划线装饰
      this.bannerData[index].className = this.bannerData[index].className += ' active';

      // 清空搜索结果数据
      this.searchResultArray = [];
      // 清空选择数据
      this.itemIdArray = [];

      // 清空选择标记状态
      this.datastructArray[index].itemList.forEach(function (item, index) {
        item.active = false;
      })

      // 取对应list数据渲染搜索词列表
      this.searchResultArray = this.datastructArray[index].itemList;
      this.searchResultDbArray = this.datastructArray[index].itemList;
      this.searchDbResultArray = this.dbItem[index];
      this.couponSelected = this.dbItem[index][0].itemId;
      this.gaojiGetJosn(this.couponSelected);
    },

    // 查找某个值在数组中的位置
    fnIndexOf: function (array, val) {
      for (var i = 0; i < array.length; i++) {
        if (array[i] == val) return i;
      }
      return -1;
    },

    // 根据固定值位置在数组中删除固定值
    fnArrayRmove: function (arr, val) {
      var index = this.fnIndexOf(arr, val);
      if (index > -1) {
        arr.splice(index, 1);
      }
    },
    initWebselectKeyWord: function (itemId) {

      this.bannerSearch = "";
      this.couponSelected = itemId;
      this.gaojiGetJosn(itemId);
      this.$refs.searchInput.focus();
    },
    // 点击选择表
    selectKeyWord: function (item, index, flag) {
      this.bannerSearch = "";
      //如果是链接，则不能被选中
      if (item.itemType == 'link') {
        //top.location.href=this.searchResultArray[index].itemUrl；
        var url = this.searchResultArray[index].itemUrl;
        window.api.openWebWin(url);
      } else {
        this.couponSelected = item.itemId;
        this.gaojiGetJosn(item.itemId);
        //支持默认查询全部
        if (flag != undefined) {
          //this.bannerSearch
          this.fnSearch(flag);
        }
        this.$refs.searchInput.focus();
      }
      /* // 如果选择词被选，取消被选状态并从选中itemId集合中删除当前ID
      if(this.searchResultArray[index].activeDef){
          this.searchResultArray[index].activeDef = false;
          this.fnArrayRmove(this.itemIdArray, item.itemId);
      }else{
          // 如果未选中当前被选次
          // 被选条件超过5个时，不允许在进行选择
          if(this.itemIdArray.length >= 5){
              this.$message({
                  showClose: true,
                  duration: 1000,
                  offset:385,
                  message: '您只能选择5个条件',
                  type: 'warning'
              })

              return false;
          }

          // 标记选择状态并把选中词语itemId放入集合当中
          this.searchResultArray[index].activeDef = true;
          this.itemIdArray.push(this.searchResultArray[index].itemId);
      }

      // 强制刷新当前搜索集合列表
      this.searchResultArray = JSON.parse(JSON.stringify(this.searchResultArray));
      // window.location.href = './home.html#/search-result?itemId='+itemId

      // 本地存储数据
      var searchResultArrayString = JSON.stringify(this.searchResultArray);
      localStorage.setItem('typeArray', searchResultArrayString); */
    },
    // 高级查询调用表头
    gaojiGetJosn: function (itemId) {
      var _this = this;
      this.getJson(itemId, function (getJsonObj) {
        var list = getJsonObj.queryItemFeild;
        var gjFeildTip = "请输入";

        // 处理数据
        list.forEach(function (item, index) {
          if (index == 0) {
            gjFeildTip = gjFeildTip + item.detail_feild_name;
          } else {
            gjFeildTip = gjFeildTip + " / " + item.detail_feild_name;
          }
        })
        _this.gjFeildTipTxt = gjFeildTip + "查询";
      });
    },
    // 表头请求json
    getJson: function (itemId, fnSuccess) {
      var _this = this;
      pajax.get(api.jsonUrl + 'config/' + itemId + '.json', {}).then(function (result) {
        var obj = result.data;
        //

        fnSuccess && fnSuccess(obj);
      })
    },

    unique: function (array) {
      var r = [];
      for (var i = 0, l = array.length; i < l; i++) {
        for (var j = i + 1; j < l; j++)
          //关键在这里
          if (JSON.stringify(array[i]) == JSON.stringify(array[j])) j = ++i;
        r.push(array[i]);
      }
      return r;
    },

    // 选择表事件
    fnSelect: function (val) {
      this.itemId = val;
      this.gaojiGetJosn(val);
    },

    // 点击搜索
    fnSearch: function (flag) {
      if (this.checks(this.bannerSearch)) {
        this.$message({
          showClose: true,
          duration: 1500,
          offset: 385,
          message: '查询内容不正确，请重新输入！',
          type: 'warning'
        });
        return;
      }
      // itemId: this.searchSelect
      this.itemIdArray = [];//解决首页选一个类型点查询，然后再换一个类型后，不断累加之前选择项的查询结果bug
      this.itemIdArray.push(this.couponSelected);

      if (this.itemIdArray.length == 0) {
        this.$message({
          showClose: true,
          duration: 0,
          offset: 385,
          message: '请重新选择其中一个库操作!',
          type: 'warning'
        });
        return;
      }

      if ((this.bannerSearch.trim() && this.couponSelected) || flag == 'linksearch' || flag == 'applinksearch') {
        /* if(!this.bannerSearch.trim()){
             this.$message({
                 showClose: true,
                 duration: 0,
                 offset:385,
                 message: '请输入关键字进行搜索',
                 type: 'warning'
             });
             return;
         }*/
        // loading
        var loading = this.$loading({
          lock: true,
          text: '数据加载中...',
          spinner: 'el-icon-loading',
          background: 'rgba(0, 0, 0, 0.7)'
        });

        // 本地存储搜索关键字
        // localStorage.setItem('searchkey', this.bannerSearch);
        // 存储类别索引
        // localStorage.setItem('bannerIndex', this.bannerIndex);

        // 存储历史搜索数据
        // this.hisArray = JSON.parse(localStorage.getItem('hisArray')) || [];
        // var hisObj = { "value": this.bannerSearch };
        // this.hisArray.push(hisObj);

        // this.hisArray = this.unique(this.hisArray);

        // localStorage.setItem('hisArray', JSON.stringify(this.hisArray));
        // return;
        // 调用搜索接口
        this.countNums(loading, flag);
      } else if (!this.bannerSearch.trim()) {
        this.$message({
          showClose: true,
          duration: 1000,
          offset: 385,
          message: '请输入关键字进行搜索',
          type: 'warning'
        })
      } else if (!this.couponSelected) {
        this.$message({
          showClose: true,
          duration: 1000,
          offset: 385,
          message: '请选择条件进行搜索',
          type: 'warning'
        })
      }
    },
    /**验证输入框是否非法字符  是非法字符 则返回true**/
    checks: function (param) {
      var regEn = /[`!@#$%^&*()_+?:";'[\]]/im,
        regCn = /[·！#￥——：；“”‘、，|《。》？、【】[\]]/im;
      if (regEn.test(param) || regCn.test(param)) {
        //alert('您输入了非法字符，请重新输入');
        return true;
      }
      // 判断小于两个字符的过滤， 比如 aa  11  22  那种
      if (param.length <= 2) {
        // 对连续字符做判断
        var regContinuous = /([a-zA-Z0-9])\1/;
        if (regContinuous.test(param)) {
          return true;
        }
      }
      // 判断内容是否为空
      if(param != ''){
        // 对单个判断
        if (param.length <= 1) {
          // 对连续字符做判断 整数过滤
          var integerFiltering = /^\+?[a-zA-Z0-9]*$/;
          if (integerFiltering.test(param)) {
            return true;
          }
        }
      }

      //验证数字
      if (!isNaN(param)) {
        if (/^(\d)\1+$/.test(param)) return true; //相同数字
      }
      //验证字母
      // if (/^[a-zA-Z]+$/.test(param)) return true; //都是字母
      this.$message.closeAll();
      return false;
    },
    //回车事件绑定
    gotoEnterEnvet: function () {
      if (event.keyCode == "13") {
        // 回车执行查询
        $('#search_button').click();
      }
    },
    goOldsearch: function () {
      this.$confirm('您好，新版数据查询系统上线试运行期间，新旧数据查询系统并行使用，给您带来的不便敬请谅解', '提示', {
        distinguishCancelAndClose: true,
        confirmButtonText: '前往',
        type: 'info'
      }).then(function () {
        window.api.openWebWin('http://app1.nmpa.gov.cn/data_nmpa/face3/dir.html');
      }).catch(function () {
      });
    },

    // 点击展开
    fnSearchResultUpDown: function () {
      this.searchResultUpDownState = !this.searchResultUpDownState;
    },

    // 高级搜搜
    firstSearch: function () {
      this.drawerShow = true;
    },

    // 高级搜索关闭
    handleClose: function () {

    },

    // 调用用户信息
    /* getLoginUserInfo(){
        var _this = this;
        pajax.hasTokenGet(api.getLoginUserInfo, {}).then(function(result){
        })
    } */
  },
  filters: {
    ellipsis: function (value, len) {
      if (!value) return '';
      if (value.length > len) {
        return value.slice(0, len) + '...'
      }
      return value
    }
  },
  components: {
    // 将组建加入组建库
    'my-header': 'url:./components/header.vue',
    'my-footer': 'url:./components/footer.vue'
  }
})