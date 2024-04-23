/******/ (() => { // webpackBootstrap
/******/ 	var __webpack_modules__ = ({

/***/ "./src/furo/assets/scripts/gumshoe-patched.js":
/*!****************************************************!*\
  !*** ./src/furo/assets/scripts/gumshoe-patched.js ***!
  \****************************************************/
/***/ (function(module, exports, __webpack_require__) {

var __WEBPACK_AMD_DEFINE_ARRAY__, __WEBPACK_AMD_DEFINE_RESULT__;/*!
 * gumshoejs v5.1.2 (patched by @pradyunsg)
 * A simple, framework-agnostic scrollspy script.
 * (c) 2019 Chris Ferdinandi
 * MIT License
 * http://github.com/cferdinandi/gumshoe
 */

(function (root, factory) {
  if (true) {
    !(__WEBPACK_AMD_DEFINE_ARRAY__ = [], __WEBPACK_AMD_DEFINE_RESULT__ = (function () {
      return factory(root);
    }).apply(exports, __WEBPACK_AMD_DEFINE_ARRAY__),
		__WEBPACK_AMD_DEFINE_RESULT__ !== undefined && (module.exports = __WEBPACK_AMD_DEFINE_RESULT__));
  } else {}
})(
  typeof __webpack_require__.g !== "undefined"
    ? __webpack_require__.g
    : typeof window !== "undefined"
    ? window
    : this,
  function (window) {
    "use strict";

    //
    // Defaults
    //

    var defaults = {
      // Active classes
      navClass: "active",
      contentClass: "active",

      // Nested navigation
      nested: false,
      nestedClass: "active",

      // Offset & reflow
      offset: 0,
      reflow: false,

      // Event support
      events: true,
    };

    //
    // Methods
    //

    /**
     * Merge two or more objects together.
     * @param   {Object}   objects  The objects to merge together
     * @returns {Object}            Merged values of defaults and options
     */
    var extend = function () {
      var merged = {};
      Array.prototype.forEach.call(arguments, function (obj) {
        for (var key in obj) {
          if (!obj.hasOwnProperty(key)) return;
          merged[key] = obj[key];
        }
      });
      return merged;
    };

    /**
     * Emit a custom event
     * @param  {String} type   The event type
     * @param  {Node}   elem   The element to attach the event to
     * @param  {Object} detail Any details to pass along with the event
     */
    var emitEvent = function (type, elem, detail) {
      // Make sure events are enabled
      if (!detail.settings.events) return;

      // Create a new event
      var event = new CustomEvent(type, {
        bubbles: true,
        cancelable: true,
        detail: detail,
      });

      // Dispatch the event
      elem.dispatchEvent(event);
    };

    /**
     * Get an element's distance from the top of the Document.
     * @param  {Node} elem The element
     * @return {Number}    Distance from the top in pixels
     */
    var getOffsetTop = function (elem) {
      var location = 0;
      if (elem.offsetParent) {
        while (elem) {
          location += elem.offsetTop;
          elem = elem.offsetParent;
        }
      }
      return location >= 0 ? location : 0;
    };

    /**
     * Sort content from first to last in the DOM
     * @param  {Array} contents The content areas
     */
    var sortContents = function (contents) {
      if (contents) {
        contents.sort(function (item1, item2) {
          var offset1 = getOffsetTop(item1.content);
          var offset2 = getOffsetTop(item2.content);
          if (offset1 < offset2) return -1;
          return 1;
        });
      }
    };

    /**
     * Get the offset to use for calculating position
     * @param  {Object} settings The settings for this instantiation
     * @return {Float}           The number of pixels to offset the calculations
     */
    var getOffset = function (settings) {
      // if the offset is a function run it
      if (typeof settings.offset === "function") {
        return parseFloat(settings.offset());
      }

      // Otherwise, return it as-is
      return parseFloat(settings.offset);
    };

    /**
     * Get the document element's height
     * @private
     * @returns {Number}
     */
    var getDocumentHeight = function () {
      return Math.max(
        document.body.scrollHeight,
        document.documentElement.scrollHeight,
        document.body.offsetHeight,
        document.documentElement.offsetHeight,
        document.body.clientHeight,
        document.documentElement.clientHeight,
      );
    };

    /**
     * Determine if an element is in view
     * @param  {Node}    elem     The element
     * @param  {Object}  settings The settings for this instantiation
     * @param  {Boolean} bottom   If true, check if element is above bottom of viewport instead
     * @return {Boolean}          Returns true if element is in the viewport
     */
    var isInView = function (elem, settings, bottom) {
      var bounds = elem.getBoundingClientRect();
      var offset = getOffset(settings);
      if (bottom) {
        return (
          parseInt(bounds.bottom, 10) <
          (window.innerHeight || document.documentElement.clientHeight)
        );
      }
      return parseInt(bounds.top, 10) <= offset;
    };

    /**
     * Check if at the bottom of the viewport
     * @return {Boolean} If true, page is at the bottom of the viewport
     */
    var isAtBottom = function () {
      if (
        Math.ceil(window.innerHeight + window.pageYOffset) >=
        getDocumentHeight()
      )
        return true;
      return false;
    };

    /**
     * Check if the last item should be used (even if not at the top of the page)
     * @param  {Object} item     The last item
     * @param  {Object} settings The settings for this instantiation
     * @return {Boolean}         If true, use the last item
     */
    var useLastItem = function (item, settings) {
      if (isAtBottom() && isInView(item.content, settings, true)) return true;
      return false;
    };

    /**
     * Get the active content
     * @param  {Array}  contents The content areas
     * @param  {Object} settings The settings for this instantiation
     * @return {Object}          The content area and matching navigation link
     */
    var getActive = function (contents, settings) {
      var last = contents[contents.length - 1];
      if (useLastItem(last, settings)) return last;
      for (var i = contents.length - 1; i >= 0; i--) {
        if (isInView(contents[i].content, settings)) return contents[i];
      }
    };

    /**
     * Deactivate parent navs in a nested navigation
     * @param  {Node}   nav      The starting navigation element
     * @param  {Object} settings The settings for this instantiation
     */
    var deactivateNested = function (nav, settings) {
      // If nesting isn't activated, bail
      if (!settings.nested || !nav.parentNode) return;

      // Get the parent navigation
      var li = nav.parentNode.closest("li");
      if (!li) return;

      // Remove the active class
      li.classList.remove(settings.nestedClass);

      // Apply recursively to any parent navigation elements
      deactivateNested(li, settings);
    };

    /**
     * Deactivate a nav and content area
     * @param  {Object} items    The nav item and content to deactivate
     * @param  {Object} settings The settings for this instantiation
     */
    var deactivate = function (items, settings) {
      // Make sure there are items to deactivate
      if (!items) return;

      // Get the parent list item
      var li = items.nav.closest("li");
      if (!li) return;

      // Remove the active class from the nav and content
      li.classList.remove(settings.navClass);
      items.content.classList.remove(settings.contentClass);

      // Deactivate any parent navs in a nested navigation
      deactivateNested(li, settings);

      // Emit a custom event
      emitEvent("gumshoeDeactivate", li, {
        link: items.nav,
        content: items.content,
        settings: settings,
      });
    };

    /**
     * Activate parent navs in a nested navigation
     * @param  {Node}   nav      The starting navigation element
     * @param  {Object} settings The settings for this instantiation
     */
    var activateNested = function (nav, settings) {
      // If nesting isn't activated, bail
      if (!settings.nested) return;

      // Get the parent navigation
      var li = nav.parentNode.closest("li");
      if (!li) return;

      // Add the active class
      li.classList.add(settings.nestedClass);

      // Apply recursively to any parent navigation elements
      activateNested(li, settings);
    };

    /**
     * Activate a nav and content area
     * @param  {Object} items    The nav item and content to activate
     * @param  {Object} settings The settings for this instantiation
     */
    var activate = function (items, settings) {
      // Make sure there are items to activate
      if (!items) return;

      // Get the parent list item
      var li = items.nav.closest("li");
      if (!li) return;

      // Add the active class to the nav and content
      li.classList.add(settings.navClass);
      items.content.classList.add(settings.contentClass);

      // Activate any parent navs in a nested navigation
      activateNested(li, settings);

      // Emit a custom event
      emitEvent("gumshoeActivate", li, {
        link: items.nav,
        content: items.content,
        settings: settings,
      });
    };

    /**
     * Create the Constructor object
     * @param {String} selector The selector to use for navigation items
     * @param {Object} options  User options and settings
     */
    var Constructor = function (selector, options) {
      //
      // Variables
      //

      var publicAPIs = {};
      var navItems, contents, current, timeout, settings;

      //
      // Methods
      //

      /**
       * Set variables from DOM elements
       */
      publicAPIs.setup = function () {
        // Get all nav items
        navItems = document.querySelectorAll(selector);

        // Create contents array
        contents = [];

        // Loop through each item, get it's matching content, and push to the array
        Array.prototype.forEach.call(navItems, function (item) {
          // Get the content for the nav item
          var content = document.getElementById(
            decodeURIComponent(item.hash.substr(1)),
          );
          if (!content) return;

          // Push to the contents array
          contents.push({
            nav: item,
            content: content,
          });
        });

        // Sort contents by the order they appear in the DOM
        sortContents(contents);
      };

      /**
       * Detect which content is currently active
       */
      publicAPIs.detect = function () {
        // Get the active content
        var active = getActive(contents, settings);

        // if there's no active content, deactivate and bail
        if (!active) {
          if (current) {
            deactivate(current, settings);
            current = null;
          }
          return;
        }

        // If the active content is the one currently active, do nothing
        if (current && active.content === current.content) return;

        // Deactivate the current content and activate the new content
        deactivate(current, settings);
        activate(active, settings);

        // Update the currently active content
        current = active;
      };

      /**
       * Detect the active content on scroll
       * Debounced for performance
       */
      var scrollHandler = function (event) {
        // If there's a timer, cancel it
        if (timeout) {
          window.cancelAnimationFrame(timeout);
        }

        // Setup debounce callback
        timeout = window.requestAnimationFrame(publicAPIs.detect);
      };

      /**
       * Update content sorting on resize
       * Debounced for performance
       */
      var resizeHandler = function (event) {
        // If there's a timer, cancel it
        if (timeout) {
          window.cancelAnimationFrame(timeout);
        }

        // Setup debounce callback
        timeout = window.requestAnimationFrame(function () {
          sortContents(contents);
          publicAPIs.detect();
        });
      };

      /**
       * Destroy the current instantiation
       */
      publicAPIs.destroy = function () {
        // Undo DOM changes
        if (current) {
          deactivate(current, settings);
        }

        // Remove event listeners
        window.removeEventListener("scroll", scrollHandler, false);
        if (settings.reflow) {
          window.removeEventListener("resize", resizeHandler, false);
        }

        // Reset variables
        contents = null;
        navItems = null;
        current = null;
        timeout = null;
        settings = null;
      };

      /**
       * Initialize the current instantiation
       */
      var init = function () {
        // Merge user options into defaults
        settings = extend(defaults, options || {});

        // Setup variables based on the current DOM
        publicAPIs.setup();

        // Find the currently active content
        publicAPIs.detect();

        // Setup event listeners
        window.addEventListener("scroll", scrollHandler, false);
        if (settings.reflow) {
          window.addEventListener("resize", resizeHandler, false);
        }
      };

      //
      // Initialize and return the public APIs
      //

      init();
      return publicAPIs;
    };

    //
    // Return the Constructor
    //

    return Constructor;
  },
);


/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/************************************************************************/
/******/ 	/* webpack/runtime/compat get default export */
/******/ 	(() => {
/******/ 		// getDefaultExport function for compatibility with non-harmony modules
/******/ 		__webpack_require__.n = (module) => {
/******/ 			var getter = module && module.__esModule ?
/******/ 				() => (module['default']) :
/******/ 				() => (module);
/******/ 			__webpack_require__.d(getter, { a: getter });
/******/ 			return getter;
/******/ 		};
/******/ 	})();
/******/
/******/ 	/* webpack/runtime/define property getters */
/******/ 	(() => {
/******/ 		// define getter functions for harmony exports
/******/ 		__webpack_require__.d = (exports, definition) => {
/******/ 			for(var key in definition) {
/******/ 				if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
/******/ 					Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
/******/ 				}
/******/ 			}
/******/ 		};
/******/ 	})();
/******/
/******/ 	/* webpack/runtime/global */
/******/ 	(() => {
/******/ 		__webpack_require__.g = (function() {
/******/ 			if (typeof globalThis === 'object') return globalThis;
/******/ 			try {
/******/ 				return this || new Function('return this')();
/******/ 			} catch (e) {
/******/ 				if (typeof window === 'object') return window;
/******/ 			}
/******/ 		})();
/******/ 	})();
/******/
/******/ 	/* webpack/runtime/hasOwnProperty shorthand */
/******/ 	(() => {
/******/ 		__webpack_require__.o = (obj, prop) => (Object.prototype.hasOwnProperty.call(obj, prop))
/******/ 	})();
/******/
/******/ 	/* webpack/runtime/make namespace object */
/******/ 	(() => {
/******/ 		// define __esModule on exports
/******/ 		__webpack_require__.r = (exports) => {
/******/ 			if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 				Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 			}
/******/ 			Object.defineProperty(exports, '__esModule', { value: true });
/******/ 		};
/******/ 	})();
/******/
/************************************************************************/
var __webpack_exports__ = {};
// This entry need to be wrapped in an IIFE because it need to be in strict mode.
(() => {
"use strict";
var __webpack_exports__ = {};
/*!*****************************************!*\
  !*** ./src/furo/assets/scripts/furo.js ***!
  \*****************************************/
__webpack_require__.r(__webpack_exports__);
/* harmony import */ var _gumshoe_patched_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./gumshoe-patched.js */ "./src/furo/assets/scripts/gumshoe-patched.js");
/* harmony import */ var _gumshoe_patched_js__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_gumshoe_patched_js__WEBPACK_IMPORTED_MODULE_0__);


////////////////////////////////////////////////////////////////////////////////
// Scroll Handling
////////////////////////////////////////////////////////////////////////////////
var tocScroll = null;
var header = null;
var lastScrollTop = window.pageYOffset || document.documentElement.scrollTop;
const GO_TO_TOP_OFFSET = 64;

// function scrollHandlerForHeader() {
//   if (Math.floor(header.getBoundingClientRect().top) == 0) {
//     header.classList.add("scrolled");
//   } else {
//     header.classList.remove("scrolled");
//   }
// }

function scrollHandlerForBackToTop(positionY) {
  if (positionY < GO_TO_TOP_OFFSET) {
    document.documentElement.classList.remove("show-back-to-top");
  } else {
    if (positionY < lastScrollTop) {
      document.documentElement.classList.add("show-back-to-top");
    } else if (positionY > lastScrollTop) {
      document.documentElement.classList.remove("show-back-to-top");
    }
  }
  lastScrollTop = positionY;
}

function scrollHandlerForTOC(positionY) {
  if (tocScroll === null) {
    return;
  }

  // top of page.
  if (positionY == 0) {
    tocScroll.scrollTo(0, 0);
  } else if (
    // bottom of page.
    Math.ceil(positionY) >=
    Math.floor(document.documentElement.scrollHeight - window.innerHeight)
  ) {
    tocScroll.scrollTo(0, tocScroll.scrollHeight);
  } else {
    // somewhere in the middle.
    const current = document.querySelector(".scroll-current");
    if (current == null) {
      return;
    }

    // https://github.com/pypa/pip/issues/9159 This breaks scroll behaviours.
    // // scroll the currently "active" heading in toc, into view.
    // const rect = current.getBoundingClientRect();
    // if (0 > rect.top) {
    //   current.scrollIntoView(true); // the argument is "alignTop"
    // } else if (rect.bottom > window.innerHeight) {
    //   current.scrollIntoView(false);
    // }
  }
}

function scrollHandler(positionY) {
  // scrollHandlerForHeader();
  scrollHandlerForBackToTop(positionY);
  scrollHandlerForTOC(positionY);
}

////////////////////////////////////////////////////////////////////////////////
// Theme Toggle
////////////////////////////////////////////////////////////////////////////////
function setTheme(mode) {
  if (mode !== "light" && mode !== "dark" && mode !== "auto") {
    console.error(`Got invalid theme mode: ${mode}. Resetting to auto.`);
    mode = "auto";
  }

  document.body.dataset.theme = mode;
  localStorage.setItem("theme", mode);
  console.log(`Changed to ${mode} mode.`);
}

function cycleThemeOnce() {
  const currentTheme = localStorage.getItem("theme") || "auto";
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

  if (prefersDark) {
    // Auto (dark) -> Light -> Dark
    if (currentTheme === "auto") {
      setTheme("light");
    } else if (currentTheme == "light") {
      setTheme("dark");
    } else {
      setTheme("auto");
    }
  } else {
    // Auto (light) -> Dark -> Light
    if (currentTheme === "auto") {
      setTheme("dark");
    } else if (currentTheme == "dark") {
      setTheme("light");
    } else {
      setTheme("auto");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Setup
////////////////////////////////////////////////////////////////////////////////
function setupScrollHandler() {
  // Taken from https://developer.mozilla.org/en-US/docs/Web/API/Document/scroll_event
  let last_known_scroll_position = 0;
  let ticking = false;

  window.addEventListener("scroll", function (e) {
    last_known_scroll_position = window.scrollY;

    if (!ticking) {
      window.requestAnimationFrame(function () {
        scrollHandler(last_known_scroll_position);
        ticking = false;
      });

      ticking = true;
    }
  });
  window.scroll();
}

function setupScrollSpy() {
  if (tocScroll === null) {
    return;
  }

  // Scrollspy -- highlight table on contents, based on scroll
  // new Gumshoe(".toc-tree a", {
  //   reflow: true,
  //   recursive: true,
  //   navClass: "scroll-current",
  //   offset: () => {
  //     let rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
  //     return header.getBoundingClientRect().height + 0.5 * rem + 1;
  //   },
  // });
}

function setupTheme() {
  // Attach event handlers for toggling themes
  const buttons = document.getElementsByClassName("theme-toggle");
  Array.from(buttons).forEach((btn) => {
    btn.addEventListener("click", cycleThemeOnce);
  });
}

function setup() {
  setupTheme();
  setupScrollHandler();
  setupScrollSpy();
}

////////////////////////////////////////////////////////////////////////////////
// Main entrypoint
////////////////////////////////////////////////////////////////////////////////
function main() {
  document.body.parentNode.classList.remove("no-js");

  // header = document.querySelectorAll("header")[1];
  tocScroll = document.querySelector(".toc-scroll");

  setup();
}

document.addEventListener("DOMContentLoaded", main);

})();

// This entry need to be wrapped in an IIFE because it need to be in strict mode.
(() => {
"use strict";
/*!******************************************!*\
  !*** ./src/furo/assets/styles/furo.sass ***!
  \******************************************/
__webpack_require__.r(__webpack_exports__);
// extracted by mini-css-extract-plugin

})();

/******/ })()
;
//# sourceMappingURL=furo.js.map
