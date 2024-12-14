import akka.actor.{ActorSystem, Props}
import akka.persistence.{PersistentActor, SnapshotOffer}

// Commands and Events
sealed trait Command
case class AddItem(item: String) extends Command
case object GetItems extends Command

sealed trait Event
case class ItemAdded(item: String) extends Event

// Define the Persistent Actor
class ShoppingCart extends PersistentActor {
  override def persistenceId: String = "shopping-cart-1"

  var items: List[String] = Nil

  def updateState(event: Event): Unit = event match {
    case ItemAdded(item) => items ::= item
  }

  def receiveCommand: Receive = {
    case AddItem(item) =>
      persist(ItemAdded(item)) { event =>
        updateState(event)
        println(s"Added item: $item")
      }
    case GetItems =>
      sender() ! items.reverse
  }

  def receiveRecover: Receive = {
    case event: Event => updateState(event)
    case SnapshotOffer(_, snapshot: List[String]) =>
      items = snapshot
  }
}

// Usage Example
object PersistentActorApp extends App {
  val system = ActorSystem("PersistentSystem")
  val cart = system.actorOf(Props[ShoppingCart](), "shoppingCart")

  cart ! AddItem("Apple")
  cart ! AddItem("Banana")
  cart ! GetItems
}